import sys
import torch
from tqdm import tqdm
sys.path.append("..")
from src.utils import *
from src.config import args
from src.kge_model import *
from src.model import *
from datetime import datetime
from src.utils import *
from torch.nn import functional as F
import gc
from src.tester import Tester


def train_and_eval(args, model, dataset, model_state_file, logger=None, device = "cpu"):

    print("Number of training tuples: {}".format(len(dataset.data["train"])))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_mrr = 0
    edge_list, rel_list = dataset.data["train_edge_graph"], dataset.data["train_rel_graph"]
    edge_list = torch.from_numpy(edge_list).to(device) 
    rel_list = torch.from_numpy(rel_list).to(device)
    
    for epoch in range(args.n_epoch):
        print("\nepoch:"+str(epoch)+ ' Time: ' + datetime.strftime(datetime.now(),'%Y-%m-%d %H:%M:%S'))
        model.train()
        was_last_batch = False
        losses = []
        iteration = 0
        if args.batch_per_epoch is not None:
            dataset.set_batch_per_epoch(args.batch_per_epoch)
            
        progress_bar = tqdm(total = dataset.num_batch(batch_size = args.batch_size, mode="train"))
        while not was_last_batch:
            batch = dataset.next_batch(batch_size=args.batch_size, neg_ratio=args.neg_ratio, mode="train", device=device)
            was_last_batch = dataset.was_last_batch(mode="train")
            targets = batch.labels.float()
            predictions = model(batch, edge_list, rel_list) # output shape: (roughly batch_size * arity, negative_num + 1)

            loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
            neg_weight = torch.ones_like(predictions)
            if args.adversarial_temperature > 0:
                with torch.no_grad():
                    neg_weight[:, 1:] = F.softmax(predictions[:, 1:] / args.adversarial_temperature, dim=-1)
            else:
                neg_weight[:, 1:] = 1 / args.neg_ratio
            loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
            loss = loss.mean()
            losses.append(loss.item())
            loss.backward()
            
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
            if ((iteration + 1) % args.accum_iter == 0) or was_last_batch:
                optimizer.step()
                optimizer.zero_grad()
            iteration+=1
            progress_bar.set_description(f"Iteration {iteration}")
            progress_bar.set_postfix(loss=loss.item())
            progress_bar.update(1)
            del batch, targets, predictions, loss
            gc.collect()
            torch.cuda.empty_cache()
        
        
            
        avg_loss = sum(losses) / len(losses)
        print("average binary cross entropy: {}".format(avg_loss))
        
        # evaluation
        if (epoch + 1) % args.eval_every == 0:
            print("valid dataset eval:")
            mrr_valid = test(model, dataset, log_mode = "valid", logger=logger, device = device)

            if mrr_valid >= best_mrr:
                best_mrr = mrr_valid
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'args': args}, model_state_file)
                print("best_mrr updated(epoch %d)!" %epoch)
        progress_bar.close()  
        del progress_bar
                
    print("\nFinal test dataset with best model:...")
    mrr_test = test(model, dataset, model_name=model_state_file, logger=logger, log_mode = "test", device = device)

    return best_mrr

    # testing
def test(model, dataset, model_name = None, logger=None, log_mode = "test", device = "cpu", test_by_arity = False):
    if log_mode == "test":
        # test mode: load parameter form file
        checkpoint = torch.load(model_name, map_location=device)
        print("\nLoad Model name: {}. Using best epoch : {}. \n\nargs:{}.".format(model_name, checkpoint['epoch'], checkpoint['args']))  # use best stat checkpoint
        print("\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\nstart test\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)

    model.eval()
    # ranking = []

    if log_mode == "test":
        edge_list, rel_list = dataset.data["test_edge_graph"], dataset.data["test_rel_graph"]
    elif log_mode == "valid" or "train":
        edge_list, rel_list = dataset.data["train_edge_graph"], dataset.data["train_rel_graph"]
    else:
        raise NotImplementedError("log_mode {} not implemented".format(log_mode))
    
    edge_list = torch.from_numpy(edge_list).to(device) 
    rel_list = torch.from_numpy(rel_list).to(device)

    tester = Tester(dataset = dataset, model = model, valid_or_test = log_mode, device = device, edge_list = edge_list, rel_list = rel_list)
    measure, _ = tester.test(test_by_arity = test_by_arity)
    mrr = measure.mrr["fil"]
    

    metrics_dict = dict()
    for metric in args.metric:
        if metric == "mr":
            score = measure.mr["fil"]
        elif metric == "mrr":
            score = measure.mrr["fil"]
        elif metric.startswith("hits@"):
            score = measure.__getattribute__(f"hit{int(metric[5:])}")["fil"]
        metrics_dict[metric] = score
    metrics_dict['time'] = datetime.strftime(datetime.now(),'%Y-%m-%d %H:%M:%S')

            
    return mrr



def main():
    working_dir, model_name, random_num = create_working_directory(args)
    set_rand_seed(args.seed)
    logger = None # Replace your logger here if wandb or neptune

    if args.test:
        model_state_file = args.model_name
    else:
        model_state_file = model_name
        
    device = get_device(args)
    # load datasets
    dataset = load_data(args.dataset, device)
    print("num_entities:", dataset.num_ent(), " num_relation:", dataset.num_rel())
    # model create
    if args.model == "HC-MPNN":
        model = HC_MPNN(
            hidden_dims = args.hidden_dim, 
            num_nodes = dataset.num_ent(), 
            num_relation = dataset.num_rel(), 
            max_arity = dataset.max_arity,
            num_layer = args.num_layer,
            dropout=args.dropout,
            norm = args.norm,
            positional_encoding = args.positional_encoding,
            initialization = args.initialization,
            short_cut = args.short_cut,
            use_triton = args.use_triton,
            dependent= args.dependent,
        )
    elif args.model == "MDistMult":
        model = MDistMult(
            dataset = dataset, 
            emb_dim = args.hidden_dim,
        )
    else:
        raise NotImplementedError("Model {} not implemented".format(args.model))
    
    if args.model_name != "":
        checkpoint = torch.load(args.model_name, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print("Load Model name: {}. Using best epoch : {}. \n\nargs:{}.".format(args.model_name, checkpoint['epoch'], checkpoint['args']))
        

    model = model.to(device)
    if args.test:
        test(model, dataset, model_name = model_state_file, logger=logger, log_mode="test",device = device, test_by_arity = args.test_by_arity)
    else:
        train_and_eval(args, model, dataset, model_state_file, logger=logger,device = device)

    sys.exit()




if __name__ == '__main__':
    main()
    


