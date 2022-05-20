import torch
import numpy as np
import tqdm
from openfl.component.aggregation_functions import Median
from openfl.interface.interactive_api.experiment import TaskInterface

#CRITERION=torch.nn.MSELoss(reduction='mean')
class Task:
    """ Create task to train a federated agent """
    @staticmethod
    def createTask(loss_fn, val_fn, num_classes, aggregation_function=Median()):
        TI = TaskInterface()
        # The Interactive API supports registering functions definied in main module or imported.
        def function_defined_in_notebook(some_parameter):
            print(f'Also I accept a parameter and it is {some_parameter}')

        #The Interactive API supports overriding of the aggregation function

        # Task interface currently supports only standalone functions.
        @TI.add_kwargs(**{'loss_fn': loss_fn, 'num_classes': num_classes})
        @TI.register_fl_task(model='model', data_loader='train_loader', \
                            device='device', optimizer='optimizer')     
        @TI.set_aggregation_function(aggregation_function)
        def train(model, train_loader, optimizer, device, loss_fn, num_classes):
            # TODO we can tune the loss functon with the aux output and apply a coeff
            """    
            The following constructions, that may lead to resource race
            is no longer needed:
            
            if not torch.cuda.is_available():
                device = 'cpu'
            else:
                device = 'cuda'        
            """
            optimizer.zero_grad()                

            print(f'\n\n TASK TRAIN GOT DEVICE {device}\n\n')
            
            #function_defined_in_notebook(some_parameter)
            
            train_loader = tqdm.tqdm(train_loader, desc="train")
            model.train()
            model.to(device)
            losses = []

            for data, target in train_loader:
                data, target = torch.tensor(data).to(device), torch.tensor(
                    target).to(device, dtype=torch.float32)

                output = model(data)["out"]
                optimizer.zero_grad()                
                loss = loss_fn.forward(output, target.long())
                
                #loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().cpu().numpy())
            
            return {'train_loss ({})'.format(str(loss_fn)): np.mean(losses),}

        @TI.add_kwargs(**{'val_fn': val_fn, 'loss_fn': loss_fn, 'num_classes': num_classes})
        @TI.register_fl_task(model='model', data_loader='val_loader', device='device')     
        def validate(model, val_loader, device, val_fn, loss_fn, num_classes):

            print(f'\n\n TASK VALIDATE GOT DEVICE {device}\n\n')
            model.eval()
            model.to(device)
            
            val_loader = tqdm.tqdm(val_loader, desc="validate")
            val_score = 0
            total_samples = 0
            losses = []

            with torch.no_grad():
                for data, target in val_loader:
                    samples = target.shape[0]
                    total_samples += samples
                    data, target = torch.tensor(data).to(device), \
                       torch.tensor(target).to(device, dtype=torch.int64)
                    
                    output = model(data)["out"]

                    loss = loss_fn.forward(output, target)
                    #loss = loss_fn(output, target)
                    val = val_fn(output, target, num_classes)

                    val_score += val.sum()
                    losses.append(loss.detach().cpu().numpy())


            return {'val_score ({})'.format(str(val_fn)): val_score, 'val_loss ({})'.format(str(loss_fn)): np.mean(losses)}
        
        return TI, validate