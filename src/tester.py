import torch
from src.measure import Measure
import tqdm

class Tester:
    def __init__(self, dataset, model, valid_or_test, edge_list, rel_list, device):
        self.device = device
        self.model = model
        self.model.eval()
        self.dataset = dataset
        self.model_name = model.name
        self.valid_or_test = valid_or_test
        self.measure = Measure()
        self.edge_list = edge_list
        self.rel_list = rel_list
        self.all_facts_as_set_of_tuples = set(dataset.positive_facts_set)

    def get_rank(self, sim_scores):
        # Assumes the test fact is the first one
        return (sim_scores >= sim_scores[0]).sum()

    def create_queries(self, fact, position):
        r, eidxs = fact[0], fact[1:]
        queries = []

        # Ensure position is within the range of eidxs
        if 0 < position <= len(eidxs):
            for i in range(1, self.dataset.num_ent()):
                # Copy the original eidxs
                new_eidxs = list(eidxs)
                # Replace the element at the specified position
                new_eidxs[position - 1] = i
                # Create a new query with the modified eidxs
                queries.append((r, *new_eidxs))

        return queries


    def add_fact_and_shred(self, fact, queries, raw_or_fil):
        if raw_or_fil == "raw":
            result = [tuple(fact)] + queries
        elif raw_or_fil == "fil":
            result = [tuple(fact)] + list(set(queries) - self.all_facts_as_set_of_tuples)
        return self.shred_facts(result)


    def test(self, test_by_arity=False):
        """
        Evaluate the given dataset and print results, either by arity or all at once
        """
        normalizer = 0
        self.measure_by_arity = {}
        self.meaddsure = Measure()

        # If the dataset is JF17K and we have test data by arity, then
        # compute test accuracies by arity and show also global result
        if test_by_arity:
            # Iterate over test sets by arity
            for cur_arity in range(2,self.dataset.max_arity+1):
                # Reset the normalizer by arity
                test_by_arity = "test_{}".format(cur_arity)
                # If the dataset does not exit, continue
                if len(self.dataset.data.get(test_by_arity, ())) == 0 :
                    print("%%%%% {} does not exist. Skipping.".format(test_by_arity))
                    continue

                print("**** Evaluating arity {} having {} samples".format(cur_arity, len(self.dataset.data[test_by_arity])))
                # Evaluate the test data for arity cur_arity
                current_measure, normalizer_by_arity =  self.eval_dataset(self.dataset.data[test_by_arity])
                
                # Sum before normalizing current_measure
                normalizer += normalizer_by_arity
                self.measure += current_measure

                # Normalize the values for the current arity and save to dict
                current_measure.normalize(normalizer_by_arity)
                self.measure_by_arity[test_by_arity] = current_measure

        else:
            # Evaluate the test data for arity cur_arity
            current_measure, normalizer = self.eval_dataset(self.dataset.data[self.valid_or_test])
            self.measure = current_measure

        # If no samples were evaluated, exit with an error
        if normalizer == 0:
            raise Exception("No Samples were evaluated! Check your test or validation data!!")

        # Normalize the global measure
        self.measure.normalize(normalizer)

        # Add the global measure (by ALL arities) to the dict
        self.measure_by_arity["ALL"] = self.measure

        # Print out results
        pr_txt = "Results for ALL ARITIES in {} set".format(self.valid_or_test)
        if test_by_arity:
            for arity in self.measure_by_arity:
                if arity == "ALL":
                    print(pr_txt)
                else:
                    print("Results for arity {}".format(arity[5:]))
                print(self.measure_by_arity[arity])
        else:
            print(pr_txt)
            print(self.measure)
        return self.measure, self.measure_by_arity

    def eval_dataset(self, dataset):
        """
        Evaluate the dataset with the given model.
        """
        # Reset normalization parameter
        settings = ["raw", "fil"]
        normalizer = 0
        # Contains the measure values for the given dataset (e.g. test for arity 2)
        current_rank = Measure()
        for i, fact in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
            arity = self.dataset.max_arity - (fact == 0).sum()
            for j in range(1, arity + 1):
                normalizer += 1
                queries = self.create_queries(fact, j)
                for raw_or_fil in settings:
                    r, eidx = self.add_fact_and_shred(fact, queries, raw_or_fil)
                    # here eidx is with shape [, max_arity]
                    if (self.model_name == "HC-MPNN"):
                        arity_transformed = torch.tensor(arity).to(self.device).expand(r.size(0))
                        r, eidx, arity_transformed = tuple(map(lambda x: x.unsqueeze(0), (r, eidx, arity_transformed)))
                        sim_scores = self.model.inference(r, eidx, arity_transformed, self.edge_list,self.rel_list).squeeze().cpu().data.numpy()
                    elif (self.model_name == "MDistMult"):
                        sim_scores = self.model.inference(r.unsqueeze(0), eidx.unsqueeze(0)).squeeze().cpu().data.numpy()
                    rank = self.get_rank(sim_scores)
                    
                    current_rank.update(rank, raw_or_fil)
                

        return current_rank, normalizer

    def shred_facts(self, tuples):
        # Assuming each tuple in tuples has the same length
        num_elements = len(tuples[0])
        shredded = [[] for _ in range(num_elements)]

        # Separate each tuple into its constituent elements
        for t in tuples:
            for i, element in enumerate(t):
                shredded[i].append(element)

        # Convert each list of elements into a tensor and move to the device
        return torch.LongTensor(shredded[0]).to(self.device), torch.stack([torch.LongTensor(shredded[i]).to(self.device) for i in range(1, num_elements)], dim = -1)

