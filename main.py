from datetime import datetime

import torch
import torch.nn as nn

from utils.parser import args
from utils import logger, Trainer, Tester
from utils import init_device, init_model, FakeLR, WarmUpCosineAnnealingLR
from dataset import Cost2100DataLoader


def main():
    logger.info('=> PyTorch Version: {}'.format(torch.__version__))

    # Environment initialization
    device, pin_memory = init_device(args.seed, args.cpu, args.gpu, args.cpu_affinity)

    # Create the data loader
    train_loader, val_loader, test_loader = Cost2100DataLoader(
        root=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=pin_memory,
        scenario=args.scenario)()

    # Define model
    model = init_model(args.cr, args.size, args.pretrained)
    model.to(device)

    # Define loss function
    criterion = nn.MSELoss().to(device)

    # Inference mode
    if args.evaluate:
        Tester(model, device, criterion)(test_loader)
        return

    # Define optimizer and scheduler
    lr_init = 1e-3 if args.scheduler == 'const' else 2e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr_init)
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr_init)

    if args.scheduler == 'const':
        scheduler = FakeLR(optimizer=optimizer)
    else:
        scheduler = WarmUpCosineAnnealingLR(optimizer=optimizer,
                                            T_max=args.epochs * len(train_loader),
                                            T_warmup=30 * len(train_loader),
                                            eta_min=5e-5)

    train_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{args.scenario}_{args.cr}_{train_timestamp}"
    # Define the training pipeline
    trainer = Trainer(model=model,
                      device=device,
                      optimizer=optimizer,
                      criterion=criterion,
                      scheduler=scheduler,
                      resume=args.resume,
                      folder_name=folder_name)

    # Start training
    trainer.loop(args.epochs, train_loader, val_loader, test_loader)

    print(f"\nTraining started at: {train_timestamp}")
    # Final testing
    print("\nFinal test!")
    loss, rho, nmse = Tester(model, device, criterion)(test_loader)
    print(f"\n=! Final test loss: {loss:.3e}"
          f"\n         test rho: {rho:.3e}"
          f"\n         test NMSE: {nmse:.3e}\n")


if __name__ == "__main__":
    main()
