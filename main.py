import torch
import argparse
from src.DatasetModule import DatasetModule
from src.TradingDataset import TradingDataset
from src.TrainModule import TrainModule
from src.Model import BaseGRUModel
import pytorch_lightning as pl
from lightning.pytorch.loggers import CometLogger
import yaml


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--config')
    args = parse.parse_args()
    with open(args.config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except:
            print("Failed to read config file")

    # dataset part:
    BATCH_SIZE: int = 5000
    data_source = 'data'
    data_name = config['data_name']
    dataset = TradingDataset(data_source=data_source, data_name=data_name)
    data_module = DatasetModule(dataset=dataset, batch_size=BATCH_SIZE)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # model part:
    model = BaseGRUModel(input_size=6, hidden_size=30, num_layers=3, with_attention=False)
    train_module = TrainModule(model=model, batch_size=BATCH_SIZE)

    # config the logger
    comet_logger = CometLogger(
        api_key='lNyK4LLQynW9EQrhnWPWfvHTk',
        project_name=config['project_name']
    )

    # START!!
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = pl.Trainer(logger=[comet_logger], max_epochs=config['max_epochs'], accelerator=device)
    trainer.fit(model=train_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    comet_logger.experiment.end()


if __name__ == '__main__':
    main()
