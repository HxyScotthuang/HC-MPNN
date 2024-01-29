## Adapted from https://github.com/ServiceNow/HypE/blob/master/dataset.py

import os
import numpy as np
import torch
import math

class CustomBatch:
    def __init__(self, batch, device="cpu"):
        self.device = device 
        self.batch = batch
        self.max_arity = batch.shape[1] - 2
        self.batch_size = batch.shape[0]
        self.r = torch.tensor(self.batch[:,:,0]).long().to(self.device)
        self.entities = torch.tensor(self.batch[:,:,1:-2]).long().to(self.device)
        self.labels = torch.tensor(self.batch[:,:, -2]).to(self.device)
        self.arities = batch[:,:,-1]
        ms = np.zeros((len(batch),len(batch[0]),self.max_arity)) 
        bs = np.ones((len(batch), len(batch[0]),self.max_arity))
        for i in range(len(batch)):
            for j in range(len(batch[0])):
                ms[i][j][0:self.arities[i][j]] = 1
                bs[i][j][0:self.arities[i][j]] = 0
        self.ms = ms
        self.bs = bs
    
    def get_fact(self):
        return self.r, self.entities
    
    def get_label(self):
        return self.labels
    
    def get_arity(self):
        return self.arities

    def get_arity_mask(self):
        return self.ms

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self):
        return torch.tensor(self.batch).to(self.device)
        
class Dataset:
    """
    This class loads the dataset and generates batches for training and testing.
    To obtain it, we get a dataset object with dict:
        dataset.data["train"] -> training data, an numpy array with shape (num_triples, max_arity + 1) of form (rel, e1, e2, e3, e4, e5, e6)
        dataset.data["test"] -> test data, an numpy array with shape (num_triples, max_arity + 1) of form (rel, e1, e2, e3, e4, e5, e6)
        dataset.data["valid"] -> validation data, an numpy array with shape (num_triples, max_arity + 1) of form (rel, e1, e2, e3, e4, e5, e6)
    """
    def __init__(self, ds_name, device="cpu", binary_dataset = False, inductive_dataset = False):
        self.name = ds_name
        self.device = device
        self.binary_dataset = binary_dataset
        self.inductive_dataset = inductive_dataset
        self.dir = os.path.join("../../data", ds_name)
        self.batch_per_epoch = None
        # id zero means no entity. Entity ids start from 1.
        self.ent2id = {"":0}
        self.rel2id = {"":0}
        self.data = {}
        print("Loading the dataset {} ....".format(ds_name)) 
        self.data["train_edge"], self.data["train_rel"] = self.read(os.path.join(self.dir, "train.txt"))


        # Load the test data
        self.data["test_edge"], self.data["test_rel"] = self.read_test(os.path.join(self.dir, "test.txt"))

       
         # Read the test files by arity, if they exist
        # If they do, then test output will be displayed by arity
        if self.name == "JF17K":
            for i in range(2,7):
                test_arity = "test_{}".format(i)
                file_path = os.path.join(self.dir, "test_{}.txt".format(i))
                self.data[test_arity+"_edge"], self.data[test_arity+"_rel"] = self.read_test(file_path)
                
        self.data["valid_edge"], self.data["valid_rel"] = self.read(os.path.join(self.dir, "valid.txt"))
        
        if self.inductive_dataset:
            self.data["aux_edge"], self.data["aux_rel"] = self.read(os.path.join(self.dir, "aux.txt"))
            self.max_arity = max([len(edge) for edge in self.data["train_edge"] + self.data["test_edge"] + self.data["valid_edge"]+ self.data["aux_edge"]])
        else:
            self.max_arity = max([len(edge) for edge in self.data["train_edge"] + self.data["test_edge"] + self.data["valid_edge"]])
        
        print("Max_arity:", self.max_arity)
        if self.binary_dataset:
            assert self.max_arity == 2, "Binary dataset must have arity 2"

        self.data["train"] = self.get_numpy_tuples(self.data["train_edge"], self.data["train_rel"])
        self.data["test"] = self.get_numpy_tuples(self.data["test_edge"], self.data["test_rel"])
        self.data["valid"] = self.get_numpy_tuples(self.data["valid_edge"], self.data["valid_rel"])

        if self.name == "JF17K":
            for i in range(2,7):
                test_arity = "test_{}".format(i)
                self.data[test_arity] = self.get_numpy_tuples(self.data[test_arity+"_edge"], self.data[test_arity+"_rel"])

        if self.inductive_dataset:
            self.data["aux"] = self.get_numpy_tuples(self.data["aux_edge"], self.data["aux_rel"])

        # Graph of numpy array we are operating on
        self.data["train_edge_graph"], self.data["train_rel_graph"] = self.data["train"][:,1:], self.data["train"][:,0]

        if self.inductive_dataset:
            self.data["test_edge_graph"], self.data["test_rel_graph"] = self.data["aux"][:,1:], self.data["aux"][:,0]
        else:
            self.data["test_edge_graph"], self.data["test_rel_graph"] = self.data["train_edge_graph"], self.data["train_rel_graph"]

        self.batch_index = {"train": 0} # batch index for training ONLY

        self.positive_facts_set = list(tuple(t) for t in self.data["train"]) \
                            + list(tuple(t) for t in self.data["test"]) \
                            + list(tuple(t) for t in self.data["valid"])

    def read(self, file_path):
        if not os.path.exists(file_path):
            print("*** {} not found. Skipping. ***".format(file_path))
            return ()
        with open(file_path, "r") as f:
            lines = f.readlines()
        tuples = []
        relations = []
        for i, line in enumerate(lines):
            rel, tuple = self.tuple2ids(line.strip().split("\t"))
            tuples.append(tuple)
            relations.append(rel)
        return tuples, relations

    def read_test(self, file_path):
        if not os.path.exists(file_path):
            print("*** {} not found. Skipping. ***".format(file_path))
            return ()
        with open(file_path, "r") as f:
            lines = f.readlines()
        tuples = []
        relations = []
        if self.name == "JF17K":
            for i, line in enumerate(lines):
                splitted = line.strip().split("\t")[1:]
                rel, tuple = self.tuple2ids(splitted)
                tuples.append(tuple)
                relations.append(rel)
        else:
            for i, line in enumerate(lines):
                splitted = line.strip().split("\t")
                rel, tuple = self.tuple2ids(splitted)
                tuples.append(tuple)
                relations.append(rel)
        return tuples, relations

    def num_ent(self):
        return len(self.ent2id)

    def num_rel(self):
        return len(self.rel2id)

    def tuple2ids(self, tuple_):
        output = []
        if self.binary_dataset:
            # read binary style with relation in between
            assert len(tuple_) == 3, "Not a binary dataset"
            for ind,t in enumerate(tuple_):
                if ind == 1:
                    rel = self.get_rel_id(t)
                else:
                    output.append(self.get_ent_id(t))
        else:
            for ind,t in enumerate(tuple_):
                if ind == 0:
                    rel= self.get_rel_id(t)
                else:
                    output.append(self.get_ent_id(t))
        return rel, output

    def get_ent_id(self, ent):
        if not ent in self.ent2id:
            self.ent2id[ent] = len(self.ent2id)
        return self.ent2id[ent]

    def get_rel_id(self, rel):
        if not rel in self.rel2id:
            self.rel2id[rel] = len(self.rel2id)
        return self.rel2id[rel]

    # transform edge_list into numpy tuples
    def get_numpy_tuple(self, edge, rel):
        return np.array([rel] + list(edge) + [0] * (self.max_arity - len(edge)))

    def get_numpy_tuples(self, edges, rels):
        return np.array([self.get_numpy_tuple(edge, rel) for edge, rel in zip(edges, rels)])


    def next_pos_batch(self, batch_size,mode):
        if self.batch_per_epoch is not None:
            # Use fast training: setting batch_per_epoch to a number
            if self.batch_index[mode] + batch_size < batch_size * self.batch_per_epoch:
                batch = self.data[mode][self.batch_index[mode]: self.batch_index[mode]+batch_size]
                self.batch_index[mode] += batch_size
            else:
                batch = self.data[mode][self.batch_index[mode]: self.batch_index[mode]+batch_size]
                np.random.shuffle(self.data[mode])
                self.batch_index[mode] = 0
        else:
            if self.batch_index[mode] + batch_size < len(self.data[mode]):
                batch = self.data[mode][self.batch_index[mode]: self.batch_index[mode]+batch_size]
                self.batch_index[mode] += batch_size
            else:
                batch = self.data[mode][self.batch_index[mode]:]
                np.random.shuffle(self.data[mode])
                self.batch_index[mode] = 0
        batch = np.append(batch, np.zeros((len(batch), 1)), axis=1).astype("int") #appending the +1 label
        batch = np.append(batch, np.zeros((len(batch), 1)), axis=1).astype("int") #appending the 0 arity
        return batch

    def next_batch(self,  neg_ratio, batch_size=1,mode="train",device="cpu"):
        pos_batch = self.next_pos_batch(batch_size, mode=mode)
        batch = self.generate_neg(pos_batch, neg_ratio, mode = mode)
        custom_batch = CustomBatch(batch = batch, device=device)
        return custom_batch
       

    def generate_neg(self, pos_batch, neg_ratio,  mode="train"):
        arities = [2+self.max_arity - (t == 0).sum() for t in pos_batch]
        assert all([arity > 1 for arity in arities]), "exists a 0 or 1 arity"
        pos_batch[:,-1] = arities
        batches = []
        if mode =="train":
            for i,c in enumerate(pos_batch):
                neg_batch = np.stack([self.neg_each(c, arities[i], target_arity, neg_ratio, mode) for target_arity in range(0,arities[i])], axis=0)
                batches.append(neg_batch)
        else:
            for i,c in enumerate(pos_batch):
                neg_batch = np.stack([self.neg_each(c, arities[i], target_arity, neg_ratio, mode) for target_arity in range(0,arities[i])], axis=0)
                batches.append(neg_batch)
        return np.concatenate(batches,axis=0)

    def neg_each(self, c, arity, a, nr, mode = "train"):
        if mode == "train":
            arr = np.repeat([c], nr + 1, axis=0)
            arr[0, -2] = 1  # Mark the first as a positive sample
            positive = c[a + 1]
            valid = False
            while not valid:
                negatives = np.random.randint(low=1, high=self.num_ent(), size=nr)
                if positive not in negatives:
                    valid = True
            arr[1:nr+1, a + 1] = negatives
        else:
            true_set = [fact[a+1] for fact in self.positive_facts_set if 
                            all([fact[i] == c[i] for i in 
                                    [x for x in range(0, arity+1) if x != a] 
                                    ])
                        ]
            negative = np.array([x for x in range(1, self.num_ent()) if x not in true_set])
            arr = np.repeat([c], len(negative) + 1, axis=0)
            arr[0, -2] = 1  # Mark the first as a positive sample
            arr[1:len(negative)+1, a + 1] = negative
        return arr

    def was_last_batch(self, mode="train"):
        return (self.batch_index[mode] == 0)

    def num_batch(self, batch_size, mode="train"):
        if self.batch_per_epoch is not None:
            return self.batch_per_epoch
        return int(math.ceil(float(len(self.data[mode])) / batch_size))

    def set_batch_per_epoch(self, batch_per_epoch):
        if batch_per_epoch >= len(self.data["train"]):
            print("Warning: batch_per_epoch is larger than the number of training data, will ignore it")
        else:
            self.batch_per_epoch = batch_per_epoch