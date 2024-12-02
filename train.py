import os
from collections import OrderedDict
import wandb
import torch as t
import datasets
from transformers import GPT2TokenizerFast
from tqdm import trange

from utils import red, bold, underline, endc, purple, yellow, blue
from model import gpt2, modelConfig, trainingConfig

t.serialization.add_safe_globals([gpt2])

MAIN = __name__ == "__main__"
device = t.device("cuda" if t.cuda.is_available() else "cpu")
print(bold, red, underline, f"device set to {device}", endc)

# Boilerplate autoregressive training code
def trained(cfg: modelConfig, targs: trainingConfig, load=None):
    n_train_iter, n_test_iter = targs.train_tokens//targs.batch_size, targs.test_tokens//targs.batch_size
    
    model = gpt2(cfg, targs)

    if isinstance(load, str): # if we're loading a model from a file
        model.load(load)
    elif isinstance(load, OrderedDict): # if we're loading a model from a state_dict
        model.load_state_dict(load)

    print(bold, purple, f"model of {sum(param.numel() for param in model.parameters()):,} params created on {device}", endc)

    wandb.init(project=targs.wandb_project, name=targs.wandb_name) # initialize wandb for visualization
    wandb.watch(model)

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2') # load in gpt2's tokenizer
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    def dataset_tokenize_func(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=cfg.n_ctx, return_tensors='pt')

    dataset_title, dataset_name  = "HuggingFaceFW/fineweb-edu", "sample-10BT" # huggingface dataset
    print(f"{yellow}loading raw dataset: {dataset_title} ({dataset_name}){endc}")
    dataset = datasets.load_dataset(dataset_title, name=dataset_name, split="train", streaming=True).map(dataset_tokenize_func, batched=True) # load the dataset
    trainloader = t.utils.data.DataLoader(dataset, batch_size=targs.batch_size) # create a dataloader for the trainset and testset
    testloader = t.utils.data.DataLoader(dataset, batch_size=targs.batch_size)
    
    print(f"{yellow}train and test dataloaders created{endc}")

    nbatch = 0
    for epoch in range(targs.epochs): # training loop
        testloss, testacc = 0, 0
        with t.inference_mode():
            testiter, testrange = iter(testloader), trange(n_test_iter, ncols=120, desc="testing. . .")
            for i in testrange: # testing loop, averaging the loss and accuracy over the test tokens
                toks = next(testiter)['input_ids'].to(device)
                logits = model.forward(toks)
                testloss += model.loss(logits, toks).item()
                testacc += model.accuracy(logits, toks).item()
            testloss /= n_test_iter
            testacc /= n_test_iter
        wandb.log({"testloss": testloss, "testacc": testacc}) # log the test loss and accuracy to wandb

        # generate a completion from the model based on a prompt to inspect its progress
        print(yellow, yap := model.yap("George Washington was", show=False), endc) 
        model.log_completion(yap)

        model.train()
        trainiter, trainrange = iter(trainloader), trange(n_train_iter, ncols=120)
        for i in trainrange: # training loop, updating weights
            toks = next(trainiter)['input_ids'].to(device)
            logits = model.forward(toks)
            loss = model.loss(logits, toks)
            model.trainstep(loss)
            nbatch += 1
            if i % 10 == 0:
                wandb.log({"trainloss": loss})
                trainrange.set_description(f"{bold+purple}[{epoch}/{targs.epochs}] {blue}loss:{loss:.4f} acc:{testacc:.6f} testloss:{testloss:.6f}")

        if targs.save_name is not None: # save the model after each epoch
            t.save(model, targs.save_name + ".pth")

    testloss, testacc = 0, 0
    with t.inference_mode(): # final test after training
        testiter, testrange = iter(testloader), trange(n_test_iter, ncols=120, desc="testing. . .")
        for i in testrange:
            toks = next(testiter)['input_ids'].to(device)
            logits = model.forward(toks)
            testloss += model.loss(logits, toks).item()
            testacc += model.accuracy(logits, toks).item()
        testloss /= n_test_iter
        testacc /= n_test_iter
    wandb.log({"testloss": testloss, "testacc": testacc})
    print(f"{bold+purple+underline}final test loss: {testloss:.6f} final test accuracy: {testacc:.6f}{endc}")

    print(yellow, yap := model.yap("Harry Potter", show=False), endc)
    model.log_completion(yap)

    wandb.finish()

    model.eval()
    return model

# (read attention implementations first)
# This is the other main contribution of the paper: we can take a model trained with normal MHA layers
# and convert it to GQA with a percentage of the original pretraining compute. This function performs
# that conversion given a path to a saved, normal pretrained MHA model.
def convert_MHA_to_GQA(model_name: str, gqa_cfg: modelConfig, uptraining_cfg: trainingConfig):
    assert gqa_cfg.attention_type == "GQA", "modelConfig must have attention_type set to 'GQA'"
    assert gqa_cfg.head_group_size != 1, "modelConfig must have head_group_size set to a value greater than 1"
    try: model = t.load(model_name+".pth")
    except FileNotFoundError: raise FileNotFoundError(f"{bold+red}model at path '{os.path.join(os.getcwd(), model_name+".pth")}' was not found. Are you running from the same folder the train script is in?{endc}")
    assert model.cfg.n_heads % gqa_cfg.head_group_size == 0, "the MHA moodel's n_heads must be divisible by the GQA model's head_group_size"
    
    grouped_query_size = (model.cfg.d_model, gqa_cfg.head_group_size, model.cfg.n_heads//gqa_cfg.head_group_size, model.cfg.d_head)

    sdict = model.state_dict()
    for layer in range(model.cfg.n_layers):
        kname, vname = f"blocks.{layer}.attn.W_K", f"blocks.{layer}.attn.W_V" # here we get the names of the weights we need to alter
        # First we reshape the projections into groups based on the requested group_head_size.
        # In GQA, every head in the group should have the same K and Q weights. We obtain this
        # shared value by averaging the K and Q weights of the heads we just grouped together.
        sdict[kname] = sdict[kname].reshape(grouped_query_size).mean(dim=1) 
        sdict[vname] = sdict[vname].reshape(grouped_query_size).mean(dim=1)
        # I didn't use biases for the k/q/v input projections so we dont need to worry about them here.

    # These modified weights are our starting point for uptraining the model with the GQA attention mechanism, letting it adjust to the modified weights.
    gqa = trained(gqa_cfg, uptraining_cfg, load=sdict)

    # we make a separate instance of the initial weights for inspection purposes after returning
    starting_model = gpt2(gqa_cfg, uptraining_cfg)
    starting_model.load_state_dict(sdict)

    return gqa

if MAIN:
    t.manual_seed(0)
    
    # pretraining a model with normal multi-head attention
    MHA_model_cfg = modelConfig(attention_type="MHA")
    MHA_pretrain_cfg = trainingConfig(
        lr=3e-4,
        batch_size=8,
        train_tokens=100_000,
        test_tokens=2_000,
        wandb_name="mha_pretrain",
        save_name="gpt2s_mha"
    )
    mha_model = trained(MHA_model_cfg, MHA_pretrain_cfg) 
    
    # converting the model to GQA attention via uptraining
    GQA_model_cfg = modelConfig(attention_type="GQA", head_group_size=4)
    GQA_uptrain_cfg = trainingConfig(
        lr=1e-4, # pretraining happens with lower learning rate
        batch_size=8,
        train_tokens=10_000, # and on far fewer tokens
        test_tokens=1_000,
        wandb_name="gpa_uptrain",
        save_name="gpt2s_gqa_uptrained"
    )
    gqa_model = convert_MHA_to_GQA("gpt2s_mha", GQA_model_cfg, GQA_uptrain_cfg)

