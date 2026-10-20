import math
import time
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from torchtext.datasets import PennTreebank
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

class PosEnc(nn.Module):
    def __init__(self,d_m,dropout=0.2,size_limit=5000):
        super(PosEnc, self).__init__()
        self.dropout =nn.Dropout(p=dropout)
        p_enc=torch.zeros(size_limit,1,d_m)
        pos=torch.arange(size_limit,dtype=torch.float).unsqueeze(1)
        divider=torch.exp(torch.arange(0,d_m,2).float()*(-math.log(10000.0)/d_m))
        p_enc[:,0,0::2]=torch.sin(pos*divider)
        p_enc[:,0,1::2]=torch.cos(pos*divider)
        self.register_buffer('p_enc',p_enc)

    def forward(self,x):
        x=x+self.p_enc[:x.size(0)]
        return self.dropout(x)
        

def gen_sqr_nxt_mask(size):
    msk=torch.triu(torch.ones(size,size),diagonal=1).bool()
    return msk

class Transformer(nn.Module):
    def __init__(self,vocab_size,d_model,nhead,nhid,nlayers,dropout=0.5):
        super(Transformer,self).__init__()
        self.model_type='Transformer'
        self.pos_encoder=PosEnc(d_model,dropout)
        layers_enc=TransformerEncoderLayer(d_model,nhead,nhid,dropout)
        self.transformer_encoder=TransformerEncoder(layers_enc,nlayers)
        self.encoder=nn.Embedding(vocab_size,d_model)
        self.d_model=d_model
        self.decoder=nn.Linear(d_model,vocab_size)
        self.init_weights()

    def init_weights(self):
        initial_rng=0.12
        self.encoder.weight.data.uniform_(-initial_rng,initial_rng)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initial_rng,initial_rng)

    def forward(self,source,mask_source):
        source=self.encoder(source)*math.sqrt(self.d_model)
        source=self.pos_encoder(source)
        op=self.transformer_encoder(source,mask_source)
        op=self.decoder(op)
        return op
    
import torch
import re
from collections import Counter
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 替代 (1)：加载数据与建词表 ====================
# 请根据你实际解压的路径修改下面这三个文件的位置
train_path = "./data/simple-examples/data/ptb.train.txt"
valid_path = "./data/simple-examples/data/ptb.valid.txt"
test_path  = "./data/simple-examples/data/ptb.test.txt"

# 1. 简单的分词函数 (平替 get_tokenizer)
def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())

# 2. 读取并分词，生成词表
print("正在读取数据并构建词表...")
with open(train_path, "r", encoding="utf-8") as f:
    train_lines = f.readlines()

# 收集所有词
all_words = []
for line in train_lines:
    all_words.extend(tokenize(line))

# 统计词频，只保留出现次数 >= 2 的词（平替 build_vocab_from_iterator）
counter = Counter(all_words)
vocab_dict = {word: i+1 for i, (word, count) in enumerate(counter.most_common()) if count >= 2}
vocab_dict["<unk>"] = 0  # 0 留给未知词

# 自定义查表函数（平替 vocabulary 对象）
def text_to_ids(lines):
    all_ids = []
    for line in lines:
        tokens = tokenize(line)
        ids = [vocab_dict.get(token, 0) for token in tokens]
        all_ids.extend(ids)
    return torch.tensor(all_ids, dtype=torch.long)

# ==================== 替代 (2)：转换张量 ====================
# 直接读取三个文件并转成张量
with open(valid_path, "r", encoding="utf-8") as f:
    valid_lines = f.readlines()
with open(test_path, "r", encoding="utf-8") as f:
    test_lines = f.readlines()

training_text = text_to_ids(train_lines)
validation_text = text_to_ids(valid_lines)
testing_text = text_to_ids(test_lines)
print(f"词表大小: {len(vocab_dict)}, 训练集长度: {len(training_text)}")

# ==================== 替代 (3)：生成批次 ====================
def gen_batches(text_dataset, batch_size):
    num_batches = text_dataset.size(0) // batch_size
    text_dataset = text_dataset[:num_batches * batch_size]
    # 核心魔法：变形并转置，让相邻的数据落在同一个 batch 里
    text_dataset = text_dataset.view(batch_size, num_batches).t().contiguous()
    return text_dataset  # 不在这一步 move 到 device，节省显存

training_batch_size = 32
evaluation_batch_size = 16

training_data = gen_batches(training_text, training_batch_size)
validation_data = gen_batches(validation_text, evaluation_batch_size)
testing_data = gen_batches(testing_text, evaluation_batch_size)

# ==================== 替代 (4)：生成序列 ====================
max_seq_len = 64

def return_batch(src, k):
    sequence_length = min(max_seq_len, len(src) - 1 - k)
    sequence_data = src[k : k + sequence_length]
    sequence_label = src[k + 1 : k + 1 + sequence_length].reshape(-1)
    return sequence_data, sequence_label

vocabulary = vocab_dict 

num_tokens =len(vocabulary)
embedding_size=256
num_hidden_params=256
num_layers=2
num_heads=2
dropout=0.25
loss_func=nn.CrossEntropyLoss()
lr=4.0
model=Transformer(num_tokens,embedding_size,num_heads,num_hidden_params,num_layers,dropout).to(device)
opt=torch.optim.SGD(model.parameters(),lr=lr)
sch=torch.optim.lr_scheduler.StepLR(opt,1.0,gamma=0.88)

def trn():
    model.train()
    loss_total=0.
    start=time.time()
    mask_source=gen_sqr_nxt_mask(max_seq_len).to(device)
    num_batches=len(training_data) // max_seq_len
    for b,j in enumerate(range(0,training_data.size(0)-1,max_seq_len)):
        train_data_batch,train_lab_batch =return_batch(training_data,j)
        train_data_batch=train_data_batch.to(device)
        train_lab_batch=train_lab_batch.to(device)
        
        sequence_length=train_data_batch.size(0)
        if sequence_length!=max_seq_len:
            mask_source=mask_source[:sequence_length,:sequence_length]
        else:
            mask_source=gen_sqr_nxt_mask(sequence_length).to(device)
        op=model(train_data_batch,mask_source)
        loss_cur=loss_func(op.view(-1,num_tokens),train_lab_batch)
        opt.zero_grad()
        loss_cur.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),0.6)
        opt.step()

        loss_total+=loss_cur.item()
        interval=100
        if b%interval==0 and b>0:
            loss_interval=loss_total/interval
            time_delta=time.time()-start
            print(f"批次: {b}/{num_batches}, 平均损失: {loss_interval:.4f}, 时间: {time_delta:.2f} 秒")
            start=time.time()
            loss_total=0.

def val(model,validation_data):
    model.eval()
    loss_total=0.
    mask_source=gen_sqr_nxt_mask(max_seq_len).to(device)
    num_batches=len(validation_data) // max_seq_len
    with torch.no_grad():
        for b,j in enumerate(range(0,validation_data.size(0)-1,max_seq_len)):
            val_data_batch,val_lab_batch=return_batch(validation_data,j)
            val_data_batch=val_data_batch.to(device)
            val_lab_batch=val_lab_batch.to(device)
            sequence_length=val_data_batch.size(0)
            if sequence_length!=max_seq_len:
                mask_source=mask_source[:sequence_length,:sequence_length]
            else:
                mask_source=gen_sqr_nxt_mask(sequence_length).to(device)
            op=model(val_data_batch,mask_source)
            op_flat=op.view(-1,num_tokens)
            loss_cur=loss_func(op_flat,val_lab_batch)
            loss_total+=loss_cur.item()*sequence_length
    return loss_total/(len(validation_data)-1)

min_val_loss=float('inf')
epo=5
best_model_path="./pth/best_transformer_model.pt"

# for e in range(1,epo+1):
#     epoch_start_time=time.time()
#     trn()
#     val_loss=val(model,validation_data)
#     print(f"第 {e} 轮验证集损失: {val_loss:.4f}, 时间: {time.time()-epoch_start_time:.2f} 秒")
#     if val_loss<min_val_loss:
#         min_val_loss=val_loss
#         torch.save(model.state_dict(),best_model_path)
#         print(f"保存最佳模型到 {best_model_path}")
#     sch.step()



id_to_word = {i: word for word, i in vocab_dict.items()}
def generate_text(model, start_text, max_len=30, temperature=0.8):
    model.eval() 
    tokens = tokenize(start_text)
    input_ids = [vocab_dict.get(token, 0) for token in tokens] 
    generated = input_ids.copy()
    with torch.no_grad():
        for _ in range(max_len):
            src = torch.tensor(generated, dtype=torch.long).unsqueeze(1).to(device)
            seq_len = src.size(0)
            mask = gen_sqr_nxt_mask(seq_len).to(device)
            output = model(src, mask)
            last_word_logits = output[-1, 0, :] 
            
            probs = torch.softmax(last_word_logits / temperature, dim=-1)
            
            next_id = torch.multinomial(probs, 1).item()
            
            generated.append(next_id)
            
            if id_to_word.get(next_id) == '.': 
                break
                
    result_words = [id_to_word.get(idx, "<unk>") for idx in generated]
    return " ".join(result_words)

state_dict = torch.load(best_model_path, map_location=device)
model.load_state_dict(state_dict)

print("\n================ 测试说话能力 ================")
prompts = ["the company", "i think that", "in the past"]
for p in prompts:
    print(f"输入: {p}")
    print(f"输出: {generate_text(model, p, max_len=20)}\n")