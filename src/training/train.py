import time
import copy
import torch
from tqdm.auto import tqdm
from ..utils.config import config
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
import torch

TRAIN = 'train'
VAL = 'val'
TEST = 'test'

def get_scheduler(optimizer, scheduler_type='one_cycle', **kwargs):
    if scheduler_type == 'one_cycle':
        return OneCycleLR(
            optimizer,
            max_lr=kwargs.get('max_lr', 0.001),  # reduced from 0.001
            epochs=config['model']['num_epochs'],
            steps_per_epoch=kwargs.get('steps_per_epoch'),
            pct_start=0.4,  # increased warmup
            div_factor=10.0,
            final_div_factor=1e3,
        )
    elif scheduler_type == 'plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.2,  # more conservative reduction
            patience=3,   # reduced patience
            min_lr=1e-6,
            verbose=True,
            threshold=1e-4  # smaller threshold for changes
        )

def evaluate_model(model, criterion, dataloader):
    """Evaluate model on given dataset"""
    model.eval()
    loss_total = 0
    dataset_size = len(dataloader.dataset)
    
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            input = torch.cat(list(images.values()), dim=2)
            outputs = model(input)
            loss = criterion(outputs.flatten(), labels.float())
            loss_total += loss.data.item()
            
            del labels, outputs, input
            torch.cuda.empty_cache()
    
    return loss_total / dataset_size

def train_model(model, criterion, optimizer, dataloaders, num_epochs=None, display_disabled=False, scheduler=None):
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

    for epoch in tqdm(range(num_epochs), disable=display_disabled, position=2, leave=True):
        for stage in [TRAIN, VAL]:
            model.train(stage == TRAIN)
            loss_epoch = 0

            with torch.set_grad_enabled(stage == TRAIN):
                for i, data in enumerate(dataloaders[stage]):
                    images, labels = data
                    input = torch.cat(list(images.values()), dim=2)
                    outputs = model(input)
                    loss = criterion(outputs.flatten(), labels.float())
                    regularized_loss = model.get_regularization_loss()
                    total_loss = loss + regularized_loss
                    

                    if stage == TRAIN:
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()

                        if isinstance(scheduler, OneCycleLR):
                            scheduler.step()

                    loss_epoch += loss.data.item()

                    del labels, outputs, input
                    torch.cuda.empty_cache()

            avg_loss = loss_epoch / dataset_sizes[stage]

            if stage == VAL and isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_loss)

            if stage == VAL:
                if avg_loss <= best_loss:
                    best_loss = avg_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                if prev_val_loss <= avg_loss:
                    trigger_cnt += 1
                else:
                    trigger_cnt = 0
                prev_val_loss = avg_loss


            if not display_disabled:
                print(f"Epoch {epoch} {stage} Loss: {avg_loss:.4f}")

        if trigger_cnt == patience:
            print(f"No improvement in {patience} epochs, stopping the training")
            break

    # Load best model and evaluate on test set
    model.load_state_dict(best_model_wts)
    test_loss = evaluate_model(model, criterion, dataloaders[TEST])
    print(f'Test Loss with best model: {test_loss:.4f}')
    
    return model, best_loss
