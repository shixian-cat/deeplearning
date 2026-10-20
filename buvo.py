import torch
from pathlib import Path
import urllib.request
import tarfile
import re
from collections import Counter

data_dir = Path("./data")
data_dir.mkdir(exist_ok=True)

# 1. 下载并解压 PTB 数据（使用 Tomas Mikolov 的源）
url = "http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz"
tgz_path = data_dir / "simple-examples.tgz"
extract_dir = data_dir / "simple-examples"

if not tgz_path.exists():
    print("正在下载数据...")
    urllib.request.urlretrieve(url, tgz_path)

if not extract_dir.exists():
    print("正在解压...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(data_dir)

# 解压后文件位于 simple-examples/data/ptb.train.txt
file_path = extract_dir / "data" / "ptb.train.txt"

# 2. 读取文本，简单分词
text = file_path.read_text(encoding="utf-8").lower()
tokens = re.findall(r"\w+|[^\w\s]", text)

# 3. 建词表
counter = Counter(tokens)
vocab = {word: i+1 for i, (word, count) in enumerate(counter.most_common()) if count >= 2}
vocab["<unk>"] = 0

# 4. 转数字 ID
ids = [vocab.get(word, vocab["<unk>"]) for word in tokens]

# 5. 转 Tensor
data_tensor = torch.tensor(ids, dtype=torch.long)
print(f"词表大小: {len(vocab)}, 文本长度: {len(data_tensor)}")