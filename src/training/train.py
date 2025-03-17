import time
import copy
import torch
from tqdm.auto import tqdm
from ..utils.config import config

TRAIN = 'Train'
VAL = 'Val' 
TEST = 'Test'

def train_model(model, criterion, optimizer, dataloaders, num_epochs=None, display_disabled=False):
    since = time.time()

    # cfg
    patience = config['model']['patience']
    num_epochs = num_epochs or config['model']['num_epochs']
    
    trigger_cnt = 0
    prev_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    dataset_sizes = dict(map(lambda item: (item[0], len(item[1].dataset)), dataloaders.items()))

    batch_num = {
        TRAIN: len(dataloaders[TRAIN]),
        VAL: len(dataloaders[VAL]),
        TEST: len(dataloaders[TEST])
    }

    for epoch in tqdm(range(num_epochs), disable=display_disabled, position=2, leave = True):
        for stage in [TRAIN, VAL]:
            model.train(stage == TRAIN)
            loss_epoch = 0

            with torch.set_grad_enabled(stage == TRAIN):
                for i, data in enumerate(dataloaders[stage]):
                    images, labels = data
                    input = torch.cat(list(images.values()), dim=2)
                    outputs = model(input)
                    loss = criterion(outputs.flatten(), labels.float())

                    if stage == TRAIN:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    loss_epoch += loss.data.item()

                    del labels, outputs, input
                    torch.cuda.empty_cache()

            avg_loss = loss_epoch / dataset_sizes[stage]

            if stage == VAL:
                if avg_loss <= best_loss:
                    best_loss = avg_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                if prev_val_loss <= avg_loss:
                    trigger_cnt += 1
                else:
                    trigger_cnt = 0
                prev_val_loss = avg_loss

        if trigger_cnt == patience:
            print(f"No improvement in {patience}, stopping the training")
            break

    model.load_state_dict(best_model_wts)
    return model, best_loss
