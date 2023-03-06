import torch.multiprocessing as mp
import argparse


def main_func(args={}):
    #############################################
    # Imports
    #############################################
    import torch
    import random
    import torch.utils.data
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms

    import torch.nn.parallel
    import torch.optim as optim

    import copy

    # custom
    import DatasetGenerators as Datasets

    from Langevin import LangevinSampler
    from Buffer import Buffer

    from VanillaNet import VanillaNet
    from Gaussian import GaussianModel

    from Trainer import Trainer

    #############################################
    # setup
    #############################################

    # set device
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # set seed
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])

    # set dataset and dataloader
    batch_size = args['batchsize']

    dimension = None
    channels = None
    image_size = None
    dataloader = None

    if args['dataset'] == 'normalfull':
        path = "./data/dfull.pth"
        dimension = 2

        dataset = Datasets.NormalDataset(device)
        dataset.load(path)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    elif args['dataset'] == 'normalcut':
        path = "./data/dcut.pth"
        dimension = 2

        dataset = Datasets.NormalCutDataset(device)
        dataset.load(path)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    elif args['dataset'] == 'MNIST':
        path = "./data"
        image_size = 64
        dimension = image_size ** 2
        channels = 1

        dataset = datasets.MNIST(
            root=path,
            train=True,
            download=False,
            transform=transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 shuffle=True, drop_last=True, pin_memory=True)

    # set model
    model = None

    if args['model'] == 'Gaussian':
        model = GaussianModel(device, dimension).to(device)
    elif args['model'] == 'VanillaNet':
        model = VanillaNet(channels, image_size).to(device)

    # preload model
    if 'modelpath' in args and not args['modelpath'] == '':
        model.load_state_dict(torch.load(args['modelpath'])['model'])

    # set optimizer
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])

    # set learning rate scheduler
    scheduler = None

    if args['hasscheduler']:
        gamma = 0.9999
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma)

    # define criterion
    def criterion(data, sample):
        loss = torch.mean(data - sample)
        return loss

    # set sampler
    langevin_step_size = args['stepsize']
    sampler = LangevinSampler(model, langevin_step_size, device)
    sampler.clipping = args['clipping']
    sampler.sampleDistance = args['sampledistance']

    # set buffer
    buffer = Buffer(next(iter(dataloader))[0][0].shape, device=device)

    # set state saving function
    filename = args['filename']

    if args['dataset'] == 'normalfull' and args['model'] == 'Gaussian':
        def save_func(model, loss, previouslist, it, epoch):
            previouslist['loss'].append(loss)
            previouslist['evaluation'].append({
                'invcov': torch.max(torch.abs(
                    model.bilinear.weight - torch.tensor([[[4/3, -2/3], [-2/3, 4/3]]], device=device))),
                'bias': torch.max(torch.abs(-model.bias-torch.tensor([0., 1.], device=device)))
            })
            torch.save({
                'model': model.state_dict(),
                'evaluation': previouslist['evaluation'],
                'loss': previouslist['loss']
            }, filename)
    else:
        def save_func(model, loss, previouslist, it, epoch):
            previouslist['loss'].append(loss)

            if epoch % 10 == 0:
                # check if not already saved for this epoch
                if len(previouslist['evaluation']) == 0 or previouslist['evaluation'][-1][0] != epoch:
                    previouslist['evaluation'].append((epoch, copy.deepcopy(model.state_dict())))

            data = {
                'model': model.state_dict(),
                'loss': previouslist['loss'],
                'model_history': previouslist['evaluation']
            }

            torch.save(data, filename)

    #############################################
    # training
    #############################################

    num_epochs = args['epoch']
    langevin_sample_count = args['burnin']
    do_print = args['output']

    trainer = Trainer(dataloader, model, optimizer, sampler, criterion, device, buffer, scheduler=scheduler)
    trainer.save_function = save_func

    if args['independentsamples']:
        trainer.batch_size = batch_size
        trainer.independent_sampling = True

    trainer.train(num_epochs, langevin_sample_count, do_print=do_print)


if __name__ == '__main__':

    # get console arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=('normalfull', 'normalcut', 'MNIST'))
    parser.add_argument('model', choices=('Gaussian', 'VanillaNet'))
    parser.add_argument('--seeds',
                        nargs="*",
                        type=int,
                        default=[200],
                        help='seed list. default: [200]')
    parser.add_argument('--batchsizes',
                        nargs="*",
                        type=int,
                        default=[256],
                        help='dataset batch size list. default: [256]')
    parser.add_argument('--lrs',
                        nargs="*",
                        type=float,
                        default=[0.01],
                        help='learning rates. default: [0.01]')
    parser.add_argument('--stepsizes',
                        nargs="*",
                        type=float,
                        default=[0.01, 0.005],
                        help='Langevin dynamics step sizes. default [0.01, 0.005]')
    parser.add_argument('--burnins',
                        nargs="*",
                        type=int,
                        default=[100],
                        help='The number of Langevin dynamics steps. default [100]')
    parser.add_argument('--epoch', type=int, default=1000, help='The number of training epoches. default 1000')
    parser.add_argument('--output', type=bool, default=False, help='Output training stats. default False')
    parser.add_argument('--modelpath', type=str, default='', help='path to preloaded model')
    parser.add_argument('--hasscheduler', type=bool, default=False, help='Learning rate scheduler. default True')
    parser.add_argument('--clipping', type=bool, default=False, help='Langevin Sampling Image Clipping. default False')
    parser.add_argument('--sampledistance', type=int, default=1, help='Langevin Sample Distance. default 1')
    parser.add_argument('--independentsamples', type=bool, default=False, help='Langevin Sample Distance. default 1')

    args = parser.parse_args()

    dataset = args.dataset
    model = args.model
    seeds = args.seeds
    epoch = args.epoch
    sampledistance = args.sampledistance
    output = args.output
    independentsamples = args.independentsamples

    modelpath = args.modelpath

    batchsizes = args.batchsizes
    lrs = args.lrs
    stepsizes = args.stepsizes
    burnins = args.burnins
    hasScheduler = args.hasscheduler
    clipping = args.clipping

    directory = ""
    if dataset == 'normalfull':
        directory = 'FullGaussian'
    elif dataset == 'normalcut':
        directory = 'CutGaussian'
    elif dataset == 'MNIST':
        directory = 'MNIST'

    plist = []

    clipstr = '_clip' if clipping else ''
    samplediststr = '_s'+str(sampledistance) if sampledistance > 1 else ''

    for seed in seeds:
        for batchsize in batchsizes:
            for lr in lrs:
                for stepsize in stepsizes:
                    for burnin in burnins:
                        filename = 'Langevin/'+directory+'/'+str(seed)+'_'+str(batchsize)+'_'+str(lr).replace('.', '')\
                                   + '_'+str(stepsize).replace('.', '')+'_'+str(burnin)\
                                   + ('_no' if hasScheduler is False else '')\
                                   + ('_pre' if not modelpath == '' else '')+clipstr+samplediststr+'.pth'
                        p = mp.Process(target=main_func, args=[{
                            'dataset': dataset,
                            'model': model,
                            'filename': filename,
                            'seed': seed,
                            'batchsize': batchsize,
                            'lr': lr,
                            'stepsize': stepsize,
                            'burnin': burnin,
                            'epoch': epoch,
                            'output': output,
                            'modelpath': modelpath,
                            'hasscheduler': hasScheduler,
                            'clipping': clipping,
                            'sampledistance': sampledistance,
                            'independentsamples': independentsamples
                        }])
                        p.start()
                        plist.append(p)

    for p in plist:
        p.join()
