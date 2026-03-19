import numpy as np
import collections
import torch
from torch.autograd import Variable
import torch.optim as optim
import os  # 添加os模块用于检查文件是否存在

import rnn

start_token = 'G'
end_token = 'E'
batch_size = 64


def process_poems1(file_name):
    """
    :param file_name:
    :return: poems_vector have two dimmention, first is the poem, the second is the word_index
    e.g. [[1,2,3,4,5,6,7,8,9,10],[9,6,3,8,5,2,7,4,1]]
    """
    poems = []
    with open(file_name, "r", encoding='utf-8') as f:
        for line in f.readlines():
            try:
                title, content = line.strip().split(':')
                content = content.replace(' ', '')
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                        start_token in content or end_token in content:
                    continue
                if len(content) < 5 or len(content) > 80:
                    continue
                content = start_token + content + end_token
                poems.append(content)
            except ValueError as e:
                print("error")
                pass
    
    # 按诗的字数排序
    poems = sorted(poems, key=lambda line: len(line))
    
    # 统计每个字出现次数
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    counter = collections.Counter(all_words)  # 统计词和词频。
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # 排序
    words, _ = zip(*count_pairs)
    words = words[:len(words)] + (' ',)
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]
    return poems_vector, word_int_map, words


def process_poems2(file_name):
    """
    :param file_name:
    :return: poems_vector have two dimmention, first is the poem, the second is the word_index
    e.g. [[1,2,3,4,5,6,7,8,9,10],[9,6,3,8,5,2,7,4,1]]
    """
    poems = []
    with open(file_name, "r", encoding='utf-8') as f:
        for line in f.readlines():
            try:
                line = line.strip()
                if line:
                    content = line.replace(' ', '').replace('，', '').replace('。', '')
                    if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                            start_token in content or end_token in content:
                        continue
                    if len(content) < 5 or len(content) > 80:
                        continue
                    content = start_token + content + end_token
                    poems.append(content)
            except ValueError as e:
                pass
    
    # 按诗的字数排序
    poems = sorted(poems, key=lambda line: len(line))
    
    # 统计每个字出现次数
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    counter = collections.Counter(all_words)  # 统计词和词频。
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # 排序
    words, _ = zip(*count_pairs)
    words = words[:len(words)] + (' ',)
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]
    return poems_vector, word_int_map, words


def generate_batch(batch_size, poems_vec, word_to_int):
    n_chunk = len(poems_vec) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size
        x_data = poems_vec[start_index:end_index]
        y_data = []
        for row in x_data:
            y = row[1:]
            y.append(row[-1])
            y_data.append(y)
        
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches


def run_training():
    # 处理数据集（可以选择使用poems.txt或tangshi.txt）
    poems_vector, word_to_int, vocabularies = process_poems1('./poems.txt')
    # poems_vector, word_to_int, vocabularies = process_poems2('./tangshi.txt')
    
    print("finish loading data")
    print("词汇表大小:", len(word_to_int))
    print("诗歌数量:", len(poems_vector))
    BATCH_SIZE = 100
    
    # 预先计算所有batch，保持顺序一致
    all_batches_inputs, all_batches_outputs = generate_batch(BATCH_SIZE, poems_vector, word_to_int)
    n_chunk = len(all_batches_inputs)
    print("总batch数量:", n_chunk)
    
    torch.manual_seed(5)
    word_embed = rnn.word_embedding(vocab_length=len(word_to_int) + 1, embedding_dim=100)
    rnn_model = rnn.RNN_model(
        batch_sz=BATCH_SIZE,
        vocab_len=len(word_to_int) + 1,
        word_embedding=word_embed,
        embedding_dim=100,
        lstm_hidden_dim=128
    )
    
    optimizer = optim.RMSprop(rnn_model.parameters(), lr=0.01)
    loss_fun = torch.nn.NLLLoss()
    
    # 记录loss用于观察趋势
    loss_history = []
    
    for epoch in range(30):
        epoch_loss = 0
        # 使用预先计算好的batch，每个epoch都用同样的顺序（或者可以打乱，但要保持对应关系）
        for batch in range(n_chunk):
            batch_x = all_batches_inputs[batch]
            batch_y = all_batches_outputs[batch]  # (batch, time_step)
            
            batch_loss = 0
            for index in range(BATCH_SIZE):
                x = np.array(batch_x[index], dtype=np.int64)
                y = np.array(batch_y[index], dtype=np.int64)
                x = Variable(torch.from_numpy(np.expand_dims(x, axis=1)))
                y = Variable(torch.from_numpy(y))
                pre = rnn_model(x)
                batch_loss += loss_fun(pre, y)
                
                if index == 0 and batch % 20 == 0:
                    _, pre_idx = torch.max(pre, dim=1)
                    print('prediction sample:', pre_idx.data.tolist()[:20])  # 只打印前20个
                    print('b_y sample       :', y.data.tolist()[:20])
                    print('*' * 30)
            
            batch_loss = batch_loss / BATCH_SIZE
            epoch_loss += batch_loss.item()
            
            if batch % 20 == 0:
                print("epoch", epoch, 'batch', batch, '/', n_chunk, "loss:", batch_loss.item())
            
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm(rnn_model.parameters(), 1)
            optimizer.step()
            
            if batch % 50 == 0:
                torch.save(rnn_model.state_dict(), './poem_generator_rnn')
                print("save model at epoch", epoch, "batch", batch)
        
        avg_epoch_loss = epoch_loss / n_chunk
        loss_history.append(avg_epoch_loss)
        print("=" * 50)
        print(f"Epoch {epoch} 完成，平均loss: {avg_epoch_loss:.4f}")
        if epoch > 0:
            print(f"loss变化: {loss_history[epoch-1]:.4f} -> {avg_epoch_loss:.4f} ({'下降' if avg_epoch_loss < loss_history[epoch-1] else '上升'})")
        print("=" * 50)
        
        # 每个epoch结束后保存模型
        torch.save(rnn_model.state_dict(), f'./poem_generator_rnn_epoch_{epoch}')
    
    # 保存最终模型
    torch.save(rnn_model.state_dict(), './poem_generator_rnn')
    print("训练完成，最终模型已保存")
    
    # 绘制loss曲线（可选，需要matplotlib）
    try:
        import matplotlib.pyplot as plt
        plt.plot(loss_history)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('training_loss.png')
        plt.show()
    except:
        print("loss历史:", [f"{l:.4f}" for l in loss_history])

def to_word(predict, vocabs):  # 预测的结果转化成汉字
    sample = np.argmax(predict)
    if sample >= len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]


def pretty_print_poem(poem):  # 令打印的结果更工整
    shige = []
    for w in poem:
        if w == start_token or w == end_token:
            break
        shige.append(w)
    
    poem_str = ''.join(shige)
    # 按句号分割打印
    poem_sentences = poem_str.split('。')
    for s in poem_sentences:
        if s != '' and len(s) > 0:
            print(s + '。')


def gen_poem(begin_word):
    # 检查模型文件是否存在
    if not os.path.exists('./poem_generator_rnn'):
        print("错误：找不到训练好的模型文件 './poem_generator_rnn'")
        print("请先运行训练模式生成模型文件")
        return begin_word + "（模型未训练）"
    
    # 加载数据（必须和训练时使用同一个数据集）
    poems_vector, word_int_map, vocabularies = process_poems1('./poems.txt')
    # poems_vector, word_int_map, vocabularies = process_poems2('./tangshi.txt')
    
    # 创建模型
    word_embed = rnn.word_embedding(vocab_length=len(word_int_map) + 1, embedding_dim=100)
    rnn_model = rnn.RNN_model(
        batch_sz=64,
        vocab_len=len(word_int_map) + 1,
        word_embedding=word_embed,
        embedding_dim=100,
        lstm_hidden_dim=128
    )
    
    # 加载训练好的模型
    rnn_model.load_state_dict(torch.load('./poem_generator_rnn'))
    rnn_model.eval()  # 设置为评估模式
    
    # 生成诗歌
    poem = begin_word
    word = begin_word
    
    while word != end_token:
        # 将当前诗歌转为索引
        input_idx = [word_int_map.get(w, 0) for w in poem]
        input_tensor = Variable(torch.from_numpy(np.array(input_idx, dtype=np.int64)))
        
        # 预测下一个字
        output = rnn_model(input_tensor, is_test=True)
        word = to_word(output.data.numpy()[0], vocabularies)
        poem += word
        
        if len(poem) > 30:
            break
    
    return poem


if __name__ == '__main__':
    import sys
    
    print("=" * 50)
    print("RNN唐诗生成器")
    print("=" * 50)
    print("请选择运行模式：")
    print("1. 训练模式（训练模型，生成poem_generator_rnn文件）")
    print("2. 生成模式（使用已有模型生成诗歌）")
    print("=" * 50)
    
    choice = input("请输入选择 (1或2): ").strip()
    
    if choice == '1':
        print("进入训练模式...")
        run_training()
    elif choice == '2':
        print("进入生成模式...")
        if os.path.exists('./poem_generator_rnn'):
            print("生成以不同字开头的诗歌：")
            print("-" * 30)
            pretty_print_poem(gen_poem("日"))
            pretty_print_poem(gen_poem("红"))
            pretty_print_poem(gen_poem("山"))
            pretty_print_poem(gen_poem("夜"))
            pretty_print_poem(gen_poem("湖"))
            pretty_print_poem(gen_poem("海"))
            pretty_print_poem(gen_poem("月"))
            pretty_print_poem(gen_poem("君"))
        else:
            print("错误：找不到模型文件，请先运行训练模式！")
    else:
        print("输入错误，请输入1或2")