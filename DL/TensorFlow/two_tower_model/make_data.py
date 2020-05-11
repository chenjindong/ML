#! encoding:utf-8
import cjdpy
import random
import numpy as np
import json
import requests
import time

video_meta_path = "../data/video_meta.txt.large"
pos_sample_weight = 0.5  # neg_sample_weight = 1- pos_sample_weight 
neg_sample_num = 3

def load_vocab(file_name):
    data = cjdpy.load_csv(file_name)
    w2id = {line[0]:int(line[1]) for line in data if len(line)==2}
    id2w = {int(line[1]):line[0] for line in data if len(line)==2}
    return w2id, id2w

def video_meta_to_id_one_sample():
    first_category2id = load_vocab("vocab/first_category_vocab.txt")
    second_category2id = load_vocab("vocab/second_category_vocab.txt")
    media2id = load_vocab("vocab/media_vocab.txt")
    tag2id = load_vocab("vocab/tag_vocab.txt")
    
    data = cjdpy.load_list("data/video_meta.txt.small")    
    res = {"first_category":[], "second_category":[], "media":[], "tag":[]}
    for line in data:
        line = json.loads(line)
        if "旅游" not in line["tags"]: continue
        tmp = line["first_category"]
        if tmp in first_category2id:
            res["first_category"].append(first_category2id[tmp])
        else:
            res["first_category"].append(first_category2id["UNK"])
        tmp = line["second_category"]
        if tmp in second_category2id:
            res["second_category"].append(second_category2id[tmp])
        else:
            res["second_category"].append(second_category2id["UNK"])
        tmp = line["media_name"]
        if tmp in media2id:
            res["media"].append(media2id[tmp])
        else:
            res["media"].append(media2id["UNK"])
        tag_id_list = []
        for tag in line["tags"].split("|"):
            if tag in tag2id:
                tag_id_list.append(tag2id[tag])
            else:
                tag_id_list.append(tag2id["UNK"])
        tag_id_list += [tag2id["PAD"]]*6
        res["tag"].append(tag_id_list[:6])
        if len(res["first_category"]) > 0:
           break
    print(res)

def visualize_dataset(file_name):
    _, id2first_category = load_vocab("../vocab/first_category_vocab.txt")
    _, id2second_category = load_vocab("../vocab/second_category_vocab.txt")
    _, id2media = load_vocab("../vocab/media_vocab.txt")
    _, id2tag = load_vocab("../vocab/tag_vocab.txt")
    
    data = cjdpy.load_csv(file_name)
    res = []
    for line in data[:5000]:
        first_category = [id2first_category[int(id)] for id in line[0].split(" ")]
        second_category = [id2second_category[int(id)] for id in line[1].split(" ")]
        media = [id2media[int(id)] for id in line[2].split(" ")]
        # tag_list = [tags for tags in line[3].split(" ")]
        tag = []
        for tag_list in line[3].split(" "):
            tag_list = [id2tag[int(id)] for id in tag_list.split("#")]
            tag.append("#".join(tag_list))
        res_one = []
        rate_discretize = line[4].split(" ")
        for i in range(len(first_category)):
            res_one.append("\t".join([first_category[i], second_category[i], media[i], tag[i], str(rate_discretize[i])]))
        res.append("\n".join(res_one))
    return res

def video_meta_to_id(file_name):
    first_category2id, _ = load_vocab("../vocab/first_category_vocab.txt")
    second_category2id, _ = load_vocab("../vocab/second_category_vocab.txt")
    media2id, _ = load_vocab("../vocab/media_vocab.txt")
    tag2id, _ = load_vocab("../vocab/tag_vocab.txt")
    print("vocab size: ", len(first_category2id), len(second_category2id), len(media2id), len(tag2id))
    
    data = cjdpy.load_list(file_name)
    # vid_set = set(cjdpy.load_list("../data/vid_set.txt"))

    res = {"first_category":[], "second_category":[], "media":[], "tag":[]}
    for line in data:
        line = json.loads(line)
        # if line["vid"] not in vid_set:
        #     continue
        tmp = line["first_category"]
        if tmp in first_category2id:
            res["first_category"].append(first_category2id[tmp])
        else:
            res["first_category"].append(first_category2id["UNK"])
        tmp = line["second_category"]
        if tmp in second_category2id:
            res["second_category"].append(second_category2id[tmp])
        else:
            res["second_category"].append(second_category2id["UNK"])
        tmp = line["media_name"]
        if tmp in media2id:
            res["media"].append(media2id[tmp])
        else:
            res["media"].append(media2id["UNK"])
        tag_id_list = []
        for tag in line["tags"].split("|"):
            if tag in tag2id:
                tag_id_list.append(tag2id[tag])
            else:
                tag_id_list.append(tag2id["UNK"])
        tag_id_list += [tag2id["PAD"]]*6
        res["tag"].append(tag_id_list[:6])
    return res

def get_train_and_eval_data_method1(file_name, save_flag=False):
    # Target的一级类目不出现在用户行为序列视频的一级类目中
    video_dict = video_meta_to_id(video_meta_path)
    print("negetive samples size: ", len(video_dict["first_category"]),len(video_dict["second_category"]),len(video_dict["media"]), len(video_dict["tag"]))

    data = cjdpy.load_csv(file_name)
    print("origin data size: ", len(data))

    first_category_feature = []
    second_category_feature = []
    tag_feature = []
    media_category_feature = []
    rate_discretize_feature = []
    weight_feature = []
    position_feature = []
    y = []
    
    for line in data:
        pos_first_category = list(map(eval, line[0].split(" ")))
        pos_second_category = list(map(eval, line[1].split(" ")))
        pos_media = list(map(eval, line[2].split(" ")))
        pos_vid_tag_list = []
        for vid_tag in line[3].split(" "):
            pos_vid_tag_list.append(list(map(eval, vid_tag.split("#"))))
        pos_rate_discretize = list(map(eval, line[4].split(" ")))
        if pos_rate_discretize[-1] == 0: continue

        first_category_feature.append(pos_first_category)
        second_category_feature.append(pos_second_category)
        media_category_feature.append(pos_media)
        tag_feature.append(pos_vid_tag_list)
        rate_discretize_feature.append(pos_rate_discretize)
        y.append(1)
        weight_feature.append(pos_sample_weight)
        position_feature.append([i for i in range(len(pos_first_category))])

        pos_first_category_set = set(pos_first_category)
        neg_sample = 0
        while neg_sample < neg_sample_num:
            idx = random.randint(0, len(video_dict["media"])-1)
            if video_dict["first_category"][idx] in pos_first_category_set:  # and neg_sample > neg_sample_num//2: # 可以有一些比较难区分的负样本
                continue 
            neg_sample += 1

            neg_first_category = pos_first_category.copy(); neg_first_category[-1] = video_dict["first_category"][idx]
            neg_second_category = pos_second_category.copy(); neg_second_category[-1] = video_dict["second_category"][idx] 
            neg_media = pos_media.copy(); neg_media[-1] = video_dict["media"][idx]
            neg_vid_tag_list = pos_vid_tag_list.copy(); neg_vid_tag_list[-1] = video_dict["tag"][idx]
            neg_rate_discretize = pos_rate_discretize.copy(); neg_rate_discretize[-1] = 0

            first_category_feature.append(neg_first_category)
            second_category_feature.append(neg_second_category)
            media_category_feature.append(neg_media)
            tag_feature.append(neg_vid_tag_list)
            rate_discretize_feature.append(neg_rate_discretize)
            y.append(0)
            weight_feature.append(1-pos_sample_weight)
            position_feature.append([i for i in range(len(pos_first_category))])

    del video_dict, data

    def save_as_file(first_category_feature, second_category_feature, tag_feature, media_category_feature):
        print("begin save file")
        data = []
        for i in range(len(first_category_feature)):
            item_str = " ".join(map(str, first_category_feature[i])) + "\t" + " ".join(map(str, second_category_feature[i])) + \
                        "\t" + " ".join(map(str, media_category_feature[i])) + "\t" + " ".join(["#".join(map(str, tag_list)) for tag_list in tag_feature[i]])
            data.append(item_str)
            print(item_str)
        cjdpy.save_lst(data, "new_dataset.txt")
    
    if save_flag:
        save_as_file(first_category_feature, second_category_feature, tag_feature, media_category_feature)

    TRAIN_EVAL_THRESOLD = int(len(y)/10*9)
    print("train and eval: ", TRAIN_EVAL_THRESOLD, len(y)-TRAIN_EVAL_THRESOLD)

    randnum = random.randint(0,100)
    random.seed(randnum); random.shuffle(first_category_feature); first_category_feature = np.array(first_category_feature)
    first_category_feature_train, first_category_feature_eval = first_category_feature[:TRAIN_EVAL_THRESOLD], first_category_feature[TRAIN_EVAL_THRESOLD:]
    del first_category_feature

    random.seed(randnum); random.shuffle(second_category_feature); second_category_feature = np.array(second_category_feature)
    second_category_feature_train, second_category_feature_eval = second_category_feature[:TRAIN_EVAL_THRESOLD], second_category_feature[TRAIN_EVAL_THRESOLD:]
    del second_category_feature

    random.seed(randnum); random.shuffle(tag_feature); tag_feature = np.array(tag_feature)
    tag_feature_feature_train, tag_feature_feature_eval = tag_feature[:TRAIN_EVAL_THRESOLD], tag_feature[TRAIN_EVAL_THRESOLD:]
    del tag_feature

    random.seed(randnum); random.shuffle(media_category_feature); media_category_feature = np.array(media_category_feature)
    media_category_feature_train, media_category_feature_eval = media_category_feature[:TRAIN_EVAL_THRESOLD], media_category_feature[TRAIN_EVAL_THRESOLD:]
    del media_category_feature

    random.seed(randnum); random.shuffle(rate_discretize_feature); rate_discretize_feature = np.array(rate_discretize_feature)
    rate_discretize_feature_train, rate_discretize_feature_eval = rate_discretize_feature[:TRAIN_EVAL_THRESOLD], rate_discretize_feature[TRAIN_EVAL_THRESOLD:]
    del rate_discretize_feature

    random.seed(randnum); random.shuffle(weight_feature); weight_feature = np.array(weight_feature, "float32")
    weight_feature_train, weight_feature_eval = weight_feature[:TRAIN_EVAL_THRESOLD], weight_feature[TRAIN_EVAL_THRESOLD:]
    del weight_feature

    position_feature = np.array(position_feature)
    position_feature_train, position_feature_eval = position_feature[:TRAIN_EVAL_THRESOLD], position_feature[TRAIN_EVAL_THRESOLD:]
    del position_feature

    random.seed(randnum); random.shuffle(y); y = np.array(y)
    y_train, y_eval = y[:TRAIN_EVAL_THRESOLD], y[TRAIN_EVAL_THRESOLD:]
    del y

    # input_fn
    train_input_x = {"first_category": first_category_feature_train,
                     "second_category": second_category_feature_train,
                     "tag": tag_feature_feature_train,
                     "media": media_category_feature_train,
                     "rate_discretize": rate_discretize_feature_train,
                     "weight": weight_feature_train,
                     "position": position_feature_train
                     }
    eval_input_x = {"first_category": first_category_feature_eval,
                     "second_category": second_category_feature_eval,
                     "tag": tag_feature_feature_eval,
                     "media": media_category_feature_eval,
                     "rate_discretize": rate_discretize_feature_eval,
                     "weight": weight_feature_eval,
                     "position": position_feature_eval
                     }

    return train_input_x, eval_input_x, y_train, y_eval

def get_train_and_eval_data(file_name, save_flag=False):
    data = cjdpy.load_csv(file_name)
    print("origin data size: ", len(data))

    first_category_feature = []
    second_category_feature = []
    tag_feature = []
    media_category_feature = []
    rate_discretize_feature = []
    weight_feature = []
    position_feature = []
    y = []
    
    for line in data:
        pos_first_category = list(map(eval, line[0].split(" ")))
        pos_second_category = list(map(eval, line[1].split(" ")))
        pos_media = list(map(eval, line[2].split(" ")))
        pos_vid_tag_list = []
        for vid_tag in line[3].split(" "):
            pos_vid_tag_list.append(list(map(eval, vid_tag.split("#"))))
        pos_rate_discretize = list(map(eval, line[4].split(" ")))
        if pos_rate_discretize[-1] == 0:
            y.append(0)
            weight_feature.append(1-pos_sample_weight)
        else:
            y.append(1)
            weight_feature.append(pos_sample_weight)

        first_category_feature.append(pos_first_category)
        second_category_feature.append(pos_second_category)
        media_category_feature.append(pos_media)
        tag_feature.append(pos_vid_tag_list)
        rate_discretize_feature.append(pos_rate_discretize)
        position_feature.append([i for i in range(len(pos_first_category))])

    del data

    def save_as_file(first_category_feature, second_category_feature, tag_feature, media_category_feature):
        print("begin save file")
        data = []
        for i in range(len(first_category_feature)):
            item_str = " ".join(map(str, first_category_feature[i])) + "\t" + " ".join(map(str, second_category_feature[i])) + \
                        "\t" + " ".join(map(str, media_category_feature[i])) + "\t" + " ".join(["#".join(map(str, tag_list)) for tag_list in tag_feature[i]])
            data.append(item_str)
            print(item_str)
        cjdpy.save_lst(data, "new_dataset.txt")
    
    if save_flag:
        save_as_file(first_category_feature, second_category_feature, tag_feature, media_category_feature)

    TRAIN_EVAL_THRESOLD = int(len(y)/10*9)
    print("train and eval: ", TRAIN_EVAL_THRESOLD, len(y)-TRAIN_EVAL_THRESOLD)

    randnum = random.randint(0,100)
    random.seed(randnum); random.shuffle(first_category_feature); first_category_feature = np.array(first_category_feature)
    first_category_feature_train, first_category_feature_eval = first_category_feature[:TRAIN_EVAL_THRESOLD], first_category_feature[TRAIN_EVAL_THRESOLD:]
    del first_category_feature

    random.seed(randnum); random.shuffle(second_category_feature); second_category_feature = np.array(second_category_feature)
    second_category_feature_train, second_category_feature_eval = second_category_feature[:TRAIN_EVAL_THRESOLD], second_category_feature[TRAIN_EVAL_THRESOLD:]
    del second_category_feature

    random.seed(randnum); random.shuffle(tag_feature); tag_feature = np.array(tag_feature)
    tag_feature_feature_train, tag_feature_feature_eval = tag_feature[:TRAIN_EVAL_THRESOLD], tag_feature[TRAIN_EVAL_THRESOLD:]
    del tag_feature

    random.seed(randnum); random.shuffle(media_category_feature); media_category_feature = np.array(media_category_feature)
    media_category_feature_train, media_category_feature_eval = media_category_feature[:TRAIN_EVAL_THRESOLD], media_category_feature[TRAIN_EVAL_THRESOLD:]
    del media_category_feature

    random.seed(randnum); random.shuffle(rate_discretize_feature); rate_discretize_feature = np.array(rate_discretize_feature)
    rate_discretize_feature_train, rate_discretize_feature_eval = rate_discretize_feature[:TRAIN_EVAL_THRESOLD], rate_discretize_feature[TRAIN_EVAL_THRESOLD:]
    del rate_discretize_feature

    random.seed(randnum); random.shuffle(weight_feature); weight_feature = np.array(weight_feature, "float32")
    weight_feature_train, weight_feature_eval = weight_feature[:TRAIN_EVAL_THRESOLD], weight_feature[TRAIN_EVAL_THRESOLD:]
    del weight_feature

    position_feature = np.array(position_feature)
    position_feature_train, position_feature_eval = position_feature[:TRAIN_EVAL_THRESOLD], position_feature[TRAIN_EVAL_THRESOLD:]
    del position_feature

    random.seed(randnum); random.shuffle(y); y = np.array(y)
    y_train, y_eval = y[:TRAIN_EVAL_THRESOLD], y[TRAIN_EVAL_THRESOLD:]
    del y

    # input_fn
    train_input_x = {"first_category": first_category_feature_train,
                     "second_category": second_category_feature_train,
                     "tag": tag_feature_feature_train,
                     "media": media_category_feature_train,
                     "rate_discretize": rate_discretize_feature_train,
                     "weight": weight_feature_train,
                     "position": position_feature_train
                     }
    eval_input_x = {"first_category": first_category_feature_eval,
                     "second_category": second_category_feature_eval,
                     "tag": tag_feature_feature_eval,
                     "media": media_category_feature_eval,
                     "rate_discretize": rate_discretize_feature_eval,
                     "weight": weight_feature_eval,
                     "position": position_feature_eval
                     }

    return train_input_x, eval_input_x, y_train, y_eval

def get_predict_data(file_name):
    data = cjdpy.load_csv(file_name)

    print("len(data): ", len(data))

    first_category_feature = []
    second_category_feature = []
    tag_feature = []
    media_category_feature = []
    rate_discretize_feature = []
    position_feature = []
    y = []

    for line in data:
        first_category_feature.append(list(map(eval, line[0].split(" "))))
        second_category_feature.append(list(map(eval, line[1].split(" "))))
        media_category_feature.append(list(map(eval, line[2].split(" "))))
        vid_tag_list = []
        for vid_tag in line[3].split(" "):
            vid_tag_list.append(list(map(eval, vid_tag.split("#"))))
        tag_feature.append(vid_tag_list)
        rate_discretize_feature.append(list(map(eval, line[4].split(" "))))
        if rate_discretize_feature[-1][-1] == 4:
            y.append(1)
        else:
            y.append(0)
        position_feature.append([i for i in range(len(first_category_feature[-1]))])

    first_category_feature = np.array(first_category_feature)
    second_category_feature = np.array(second_category_feature)
    tag_feature = np.array(tag_feature)
    media_category_feature = np.array(media_category_feature)
    rate_discretize_feature = np.array(rate_discretize_feature)
    position_feature = np.array(position_feature)
    y = np.array(y)

    # input_fn
    input_x = {"first_category": first_category_feature,
               "second_category": second_category_feature,
               "tag": tag_feature,
               "media": media_category_feature,
               "rate_discretize": rate_discretize_feature,
               "position": position_feature
               }
    return input_x, y

# def get_predict_data_serving(file_name, num_list, kandian_today):
def get_predict_data_serving(data, num, kandian_today):

    # kandian_today = video_meta_to_id("../data/kandian_today.json")

    # data = cjdpy.load_csv(file_name)
    # print("len(data): ", len(data))

    first_category_feature = []
    second_category_feature = []
    tag_feature = []
    media_category_feature = []
    rate_discretize_feature = []
    position_feature = []
    y = []
   
    # for num in num_list:
    line = data[num]  # 指定用户行为序列是pred.txt中的第num行
    # print(line)
    first_category = list(map(eval, line[0].split(" ")))
    second_category = list(map(eval, line[1].split(" ")))
    media_category = list(map(eval, line[2].split(" ")))
    rate_discretize = list(map(eval, line[4].split(" ")))
    vid_tag_list = []
    for vid_tag in line[3].split(" "):
        vid_tag_list.append(list(map(eval, vid_tag.split("#"))))
    position = [i for i in range(len(first_category))]

    for i in range(len(kandian_today["first_category"])):
        first_category_one = first_category.copy(); first_category_one[-1] = kandian_today["first_category"][i]
        second_category_one = second_category.copy(); second_category_one[-1] = kandian_today["second_category"][i]
        media_category_one = media_category.copy(); media_category_one[-1] = kandian_today["media"][i]
        vid_tag_list_one = vid_tag_list.copy(); vid_tag_list_one[-1] = kandian_today["tag"][i]
        position_one = position.copy()
        rate_discretize_one = rate_discretize.copy()

        first_category_feature.append(first_category_one)
        second_category_feature.append(second_category_one)
        media_category_feature.append(media_category_one)
        tag_feature.append(vid_tag_list_one)
        rate_discretize_feature.append(rate_discretize_one)
        position_feature.append(position_one)

    # input_fn
    input_x = {"first_category": first_category_feature,
               "second_category": second_category_feature,
               "tag": tag_feature,
               "media": media_category_feature,
               "rate_discretize": rate_discretize_feature,
               "position": position_feature
               }
    return input_x, y

def get_vid_embedding():
    # 离线计算kandian_today中视频的embedding
    print("get vid embedding begin")
    kandian_today = video_meta_to_id("../data/kandian_today.json")
    pred_data = cjdpy.load_csv("../data/pred.txt")

    input_x, _ = get_predict_data_serving(pred_data, 1, kandian_today)  # use behavior seq 随意选择即可

    kandian = cjdpy.load_list("../data/kandian_today.json")
    kandian_str = []
    for line in kandian:
        line = json.loads(line)
        kandian_str.append(" ".join([line["first_category"], line["second_category"], line["tags"], line["media_name"]])) 

    idx_rate = []
    cmsid_embed = []
    cmsid_set = set()
    cmsid_list = []
    bad_case = 0
    for i in range(len(input_x["first_category"])):
        dict_data = {"instances": [{"first_category": input_x["first_category"][i],
                                    "second_category": input_x["second_category"][i],
                                    "tag": input_x["tag"][i],
                                    "media": input_x["media"][i],
                                    "rate_discretize": input_x["rate_discretize"][i],
                                    "position": input_x["position"][i]
                                    }]}

        try: 
            resp = requests.post('http://localhost:8515/v1/models/ttm:predict', json=dict_data)
            res = json.loads(resp.text)
            idx_rate.append([i] + res["predictions"][0]["ie"])
            cmsid_val = json.loads(kandian[i])["cmsid"]
            if cmsid_val in cmsid_set: continue
            cmsid_embed.append(res["predictions"][0]["ie"])
            cmsid_set.add(cmsid_val)
            cmsid_list.append(cmsid_val)
        except:
            bad_case += 1
        # if "predictions" not in res:
        #     bad_case += 1
        #     continue

        if i % 5000 == 0: 
            print("process", i)
    
    print("#fail to request tf serving", bad_case)

    cjdpy.save_csv(idx_rate, "ie.txt")
    cjdpy.save_csv(cmsid_embed, "cmsid_embedding.txt", " ")
    cjdpy.save_lst(cmsid_list, "cmsid.txt")
    print("get vid embedding done")

def faiss_retrieve(input_x, kandian, ie):
    # 模拟faiss，根据user embedding找相似的targe video
    # kandian = cjdpy.load_list("../data/kandian_today.json")
    kandian_str = []
    for line in kandian:
        line = json.loads(line)
        kandian_str.append("\t".join([line["first_category"], line["second_category"], line["media_name"], line["tags"]])) 
    # print("kandian:", len(kandian_str))
    # ie = cjdpy.load_csv("ie.txt")

    dict_data = {"instances": [{"first_category": input_x["first_category"][0],
                                    "second_category": input_x["second_category"][0],
                                    "tag": input_x["tag"][0],
                                    "media": input_x["media"][0],
                                    "rate_discretize": input_x["rate_discretize"][0],
                                    "position": input_x["position"][0]
                                    }]}

    resp = requests.post('http://localhost:8515/v1/models/ttm:predict', json=dict_data)
    res = json.loads(resp.text)

    if "predictions" not in res:
        return []
    
    ue = res["predictions"][0]["ue"]
    idx_rate = []
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    for i in range(len(ie)):
        tmp = np.sum(np.array(ue)*np.array(list(map(eval, ie[i][1:]))))
        prob = sigmoid(tmp)
        idx_rate.append([int(ie[i][0]), prob])

    idx_rate = sorted(idx_rate, key=lambda x:x[1], reverse=True)
    res = []
    for i in range(10):
        print(idx_rate[i], kandian_str[idx_rate[i][0]])
        res.append(kandian_str[idx_rate[i][0]])
    return res

def cal_MAP(ground_truth, predict):
    if len(ground_truth) != len(predict):
        return -1
    map_val = 0
    for i in range(len(ground_truth)):
        intersect = len(set(ground_truth[:i+1])&set(predict[:i+1]))
        map_val += 1.0*intersect/(i+1)
    return map_val/len(ground_truth)

def cal_map_metric():
    data = cjdpy.load_csv("../data/metric_data.txt")

    FUTRUE_SEQ_LEN = 10
    HISTORY_SEQ_LEN = 25
    map_score, map_fc_score, map_sc_score, map_tag_score = 0, 0, 0, 0
    eval_sample_num = 100

    # uin, first_category, second_category, media, tags, rate, rate_discretize
    for case, line in enumerate(data):
        first_category = list(map(eval, line[1].split(" ")))
        second_category = list(map(eval, line[2].split(" ")))
        media = list(map(eval, line[3].split(" ")))
        rate = list(map(eval, line[5].split(" ")))
        rate_discretize = list(map(eval, line[6].split(" ")))
        vid_tag_list = []
        for vid_tag in line[4].split(" "):
            vid_tag_list.append(list(map(eval, vid_tag.split("#"))))
        if len(first_category) < 35: 
            continue
        gt = [[i, rate[i+HISTORY_SEQ_LEN]] for i in range(FUTRUE_SEQ_LEN)]
        gt = sorted(gt, key=lambda x: x[1], reverse=True)
        gt_rank = [gt[i][0] for i in range(FUTRUE_SEQ_LEN)]

        gt_fc, gt_sc, gt_tag = {}, {}, {}
        for i in range(FUTRUE_SEQ_LEN):
            if first_category[i] not in gt_fc:
                gt_fc[first_category[i]] = 0
            gt_fc[first_category[i]] += rate[i+HISTORY_SEQ_LEN]
            if second_category[i] not in gt_sc:
                gt_sc[second_category[i]] = 0
            gt_sc[second_category[i]] += rate[i+HISTORY_SEQ_LEN]
            for tag in vid_tag_list[i]:
                if tag not in gt_tag:
                    gt_tag[tag] = 0
                gt_tag[tag] += rate[i+HISTORY_SEQ_LEN]
        
        gt_fc = sorted(gt_fc.items(), key=lambda x: x[1], reverse=True)
        gt_fc_rank = [gt_fc[i][0] for i in range(len(gt_fc))]
        gt_sc = sorted(gt_sc.items(), key=lambda x: x[1], reverse=True)
        gt_sc_rank = [gt_sc[i][0] for i in range(len(gt_sc))]
        gt_tag = sorted(gt_tag.items(), key=lambda x: x[1], reverse=True)
        gt_tag_rank = [gt_tag[i][0] for i in range(len(gt_tag))]
        # print(gt_fc_rank)

        # pd = []
        pred_rate = []
        for i in range(FUTRUE_SEQ_LEN):
            dict_data = {"instances": [{"first_category": first_category[i:i+26],
                                    "second_category": second_category[i:i+26],
                                    "tag": vid_tag_list[i:i+26],
                                    "media": media[i:i+26],
                                    "rate_discretize": rate_discretize[i:i+26],
                                    "position": [i for i in range(26)]
                                    }]}

            resp = requests.post('http://localhost:8515/v1/models/ttm:predict', json=dict_data)
            res = json.loads(resp.text)
            # pd.append([i, res["predictions"][0]["y"]])
            pred_rate.append([i, res["predictions"][0]["y"]])
        pd = sorted(pred_rate, key=lambda x: x[1], reverse=True)
        pd_rank = [pd[i][0] for i in range(FUTRUE_SEQ_LEN)]
        map_score += cal_MAP(pd_rank, gt_rank)

        pd_fc, pd_sc, pd_tag = {}, {}, {}
        for i in range(FUTRUE_SEQ_LEN):
            if first_category[i] not in pd_fc:
                pd_fc[first_category[i]] = 0
            pd_fc[first_category[i]] += pred_rate[i][1]
            if second_category[i] not in pd_sc:
                pd_sc[second_category[i]] = 0
            pd_sc[second_category[i]] += pred_rate[i][1]
            for tag in vid_tag_list[i]:
                if tag not in pd_tag:
                    pd_tag[tag] = 0
                pd_tag[tag] += pred_rate[i][1]
            
        pd_fc = sorted(pd_fc.items(), key=lambda x: x[1], reverse=True)
        pd_fc_rank = [pd_fc[i][0] for i in range(len(pd_fc))]
        map_fc_score += cal_MAP(pd_fc_rank, gt_fc_rank)
        # print(pd_fc_rank)
        # break
        pd_sc = sorted(pd_sc.items(), key=lambda x: x[1], reverse=True)
        pd_sc_rank = [pd_sc[i][0] for i in range(len(pd_sc))]
        map_sc_score += cal_MAP(pd_sc_rank, gt_sc_rank)

        pd_tag = sorted(pd_tag.items(), key=lambda x: x[1], reverse=True)
        pd_tag_rank = [pd_tag[i][0] for i in range(len(pd_tag))]
        map_tag_score += cal_MAP(pd_tag_rank, gt_tag_rank)

        if case > eval_sample_num: break
    print("MAP for video score: ", map_score/eval_sample_num)
    print("MAP for video first category score: ", map_fc_score/eval_sample_num)
    print("MAP for video second category score: ", map_sc_score/eval_sample_num)
    print("MAP for video tag score: ", map_tag_score/eval_sample_num)

def cal_diversity_metric():
    pred_data_vis = visualize_dataset("../data/pred.txt")
    pred_data = cjdpy.load_csv("../data/pred.txt")
    kandian_today_id = video_meta_to_id("../data/kandian_today.json")
    kandian_today = cjdpy.load_list("../data/kandian_today.json")

    ie = cjdpy.load_csv("ie.txt")

    test_sample_num = 10
    pred_idx = [i*30 for i in range(test_sample_num)]
    fc_cnt, sc_cnt, media_cnt, tag_cnt = 0, 0, 0, 0
    total_eval_video = 0
    for num in pred_idx:
        print(pred_data_vis[num])
        fc_set, sc_set, media_set, tag_set = set(), set(), set(), set()
        for line in pred_data_vis[num].split("\n"):
            items = line.split("\t")
            fc_set.add(items[0])
            sc_set.add(items[1])
            media_set.add(items[2])
            tag_set = tag_set | set(items[3].split("#"))
        input_x, _ = get_predict_data_serving(pred_data, num, kandian_today_id)
        topK = faiss_retrieve(input_x, kandian_today, ie)
        # print(topK)
        total_eval_video += len(topK)
        for line in topK:
            items = line.split("\t")
            # print(items)
            if items[0] not in fc_set: fc_cnt += 1
            if items[1] not in sc_set: sc_cnt += 1
            if items[2] not in media_set: media_cnt += 1
            for tag in items[3].split("|"):
                if tag not in tag_set:
                    tag_cnt += 1
        print()
        # break
    print("first_category", test_sample_num*fc_cnt/total_eval_video)
    print("second_category", test_sample_num*sc_cnt/total_eval_video)
    print("media", test_sample_num*media_cnt/total_eval_video)
    print("tag", test_sample_num*tag_cnt/total_eval_video)


if __name__ =="__main__":
    # one
    # get_train_and_eval_data("../data/dataset.txt.small", True)
    # visualize_dataset("new_dataset.txt")
    # video_meta_to_id_one_sample()
    
    # two
    # pred_data = visualize_dataset("../data/pred.txt") # 前5k个
    # num = 100
    # print(pred_data[num])

    
    #cal_map_metric()
    #get_vid_embedding()
    cal_diversity_metric()




