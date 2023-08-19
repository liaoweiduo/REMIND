import argparse
import torch
import json
import os
import copy
from image_classification_experiments.REMINDModel import REMINDModel
from image_classification_experiments.imagenet_base_initialization import *

from torch.utils.data import DataLoader, ConcatDataset

torch.multiprocessing.set_sharing_strategy('file_system')


def get_data_loader(images_dir, label_dir, split, min_class, max_class, batch_size=128, return_item_ix=False,
                    target='CGQA', fewshot_mode='sys'):
    if target == 'CGQA':
        data_loader = get_cgqa_data_loader(images_dir, label_dir, split, min_class, max_class, batch_size=batch_size,
                                           return_item_ix=return_item_ix, fewshot_mode=fewshot_mode)
    elif target == 'COBJ':
        data_loader = get_cobj_data_loader(images_dir, label_dir, split, min_class, max_class, batch_size=batch_size,
                                           return_item_ix=return_item_ix, fewshot_mode=fewshot_mode)
    else:
        data_loader = utils_imagenet.get_imagenet_data_loader(images_dir + '/' + split, label_dir, split,
                                                              batch_size=batch_size, shuffle=False, min_class=min_class,
                                                              max_class=max_class, return_item_ix=return_item_ix)
    return data_loader


def get_cgqa_data_loader(images_dir, label_dir, split, min_class, max_class, batch_size=128, return_item_ix=False,
                         fewshot_mode='sys'):
    from image_classification_experiments.cgqa import continual_training_benchmark, fewshot_testing_benchmark

    '''from min_class depends which task to use'''
    task_id_from = min_class // 10
    task_id_to = max_class // 10    # do not contain the last
    if task_id_from < 10:
        benchmark = continual_training_benchmark(
            n_experiences=10,
            image_size=(224, 224),
            return_task_id=True,
            label_from_zero_each_task=False,
            seed=1234, shuffle=True,
            dataset_root=images_dir)

        if split == 'train':
            return DataLoader(benchmark.train_datasets[task_id_from], batch_size=batch_size, shuffle=True,
                              num_workers=10)
        elif split == 'val':
            if task_id_from == task_id_to:
                cat_dataset = benchmark.test_datasets[task_id_from]
            else:
                cat_dataset = ConcatDataset(benchmark.test_datasets[task_id_from:task_id_to])
            return DataLoader(cat_dataset, batch_size=batch_size, shuffle=False, num_workers=10)

    else:
        num_tasks = 50
        benchmark = fewshot_testing_benchmark(
            n_experiences=num_tasks, image_size=(224, 224),
            mode=fewshot_mode,
            n_way=10, n_shot=10, n_val=5, n_query=10,
            task_offset=10, class_offset=100,       # task all as 10, class all as 100-109
            seed=1234,
            dataset_root=images_dir)

        train_loaders = [DataLoader(benchmark.train_datasets[task_id],
                                    batch_size=100, shuffle=True, num_workers=10)
                         for task_id in range(num_tasks)]
        test_loaders = [DataLoader(benchmark.test_datasets[task_id],
                                   batch_size=100, shuffle=False, num_workers=10)
                        for task_id in range(num_tasks)]

        return train_loaders, test_loaders


def get_cobj_data_loader(images_dir, label_dir, split, min_class, max_class, batch_size=128, return_item_ix=False,
                         fewshot_mode='sys'):
    from image_classification_experiments.cobj import continual_training_benchmark, fewshot_testing_benchmark

    '''from min_class depends which task to use'''
    task_id_from = min_class // 10
    task_id_to = max_class // 10    # do not contain the last
    if task_id_from < 3:
        benchmark = continual_training_benchmark(
            n_experiences=3,
            image_size=(224, 224),
            return_task_id=True,
            label_from_zero_each_task=False,
            seed=1234, shuffle=True,
            dataset_root=images_dir)

        if split == 'train':
            return DataLoader(benchmark.train_datasets[task_id_from], batch_size=batch_size, shuffle=True,
                              num_workers=10)
        elif split == 'val':
            if task_id_from == task_id_to:
                cat_dataset = benchmark.test_datasets[task_id_from]
            else:
                cat_dataset = ConcatDataset(benchmark.test_datasets[task_id_from:task_id_to])
            return DataLoader(cat_dataset, batch_size=batch_size, shuffle=False, num_workers=10)

    else:
        num_tasks = 50
        benchmark = fewshot_testing_benchmark(
            n_experiences=num_tasks, image_size=(224, 224),
            mode=fewshot_mode,
            n_way=10, n_shot=10, n_val=5, n_query=10,
            task_offset=3, class_offset=30,       # task all as 3, class all as 30-39
            seed=1234,
            dataset_root=images_dir)

        train_loaders = [DataLoader(benchmark.train_datasets[task_id],
                                    batch_size=100, shuffle=True, num_workers=10)
                         for task_id in range(num_tasks)]
        test_loaders = [DataLoader(benchmark.test_datasets[task_id],
                                   batch_size=100, shuffle=False, num_workers=10)
                        for task_id in range(num_tasks)]

        return train_loaders, test_loaders


def compute_accuracies(loader, remind, pq):
    _, probas, y_test_init = remind.predict(loader, pq)
    top1, top5 = utils.accuracy(probas, y_test_init, topk=(1, 5))
    return probas, top1, top5


def update_accuracies(args, curr_max_class, remind, pq, accuracies, test_loader=None, mode='sys', save=False):
    if test_loader is not None:
        probas, top1, top5 = compute_accuracies(test_loader, remind, pq)
        accuracies[f'{mode}_top1'].append(float(top1))
        if save:
            utils.save_accuracies(accuracies, min_class_trained=args.min_class, max_class_trained=curr_max_class,
                                  save_path=args.save_dir)
    else:
        base_test_loader = get_data_loader(args.images_dir, args.label_dir, 'val', min_class=args.min_class,
                                           max_class=args.base_init_classes, target=args.datasets)
        base_probas, base_top1, base_top5 = compute_accuracies(base_test_loader, remind, pq)
        print('\nBase Init Classes (%d-%d): top1=%0.2f%% -- top5=%0.2f%%' % (
            args.min_class, args.base_init_classes - 1, base_top1, base_top5))

        non_base_classes_test_loader = get_data_loader(args.images_dir, args.label_dir, 'val', args.base_init_classes,
                                                       curr_max_class, target=args.datasets)
        non_base_probas, non_base_top1, non_base_top5 = compute_accuracies(non_base_classes_test_loader, remind, pq)

        print('Non-Base Init Classes (%d-%d): top1=%0.2f%% -- top5=%0.2f%%' % (
            args.base_init_classes, curr_max_class - 1, non_base_top1, non_base_top5))

        seen_classes_test_loader = get_data_loader(args.images_dir, args.label_dir, 'val',
                                                   args.min_class, curr_max_class, target=args.datasets)
        seen_probas, seen_top1, seen_top5 = compute_accuracies(seen_classes_test_loader, remind, pq)
        print('All Seen Classes (%d-%d): top1=%0.2f%% -- top5=%0.2f%%' % (
            args.min_class, curr_max_class - 1, seen_top1, seen_top5))

        accuracies['base_classes_top1'].append(float(base_top1))
        accuracies['base_classes_top5'].append(float(base_top5))
        accuracies['non_base_classes_top1'].append(float(non_base_top1))
        accuracies['non_base_classes_top5'].append(float(non_base_top5))
        accuracies['seen_classes_top1'].append(float(seen_top1))
        accuracies['seen_classes_top5'].append(float(seen_top5))

        utils.save_accuracies(accuracies, min_class_trained=args.min_class, max_class_trained=curr_max_class,
                              save_path=args.save_dir)
        utils.save_predictions(seen_probas, args.min_class, curr_max_class, args.save_dir)


def streaming(args, remind):
    accuracies = {'base_classes_top1': [], 'non_base_classes_top1': [], 'seen_classes_top1': [],
                  'base_classes_top5': [], 'non_base_classes_top5': [], 'seen_classes_top5': [],
                  'sys_top1': [], 'pro_top1': [], 'sub_top1': [], 'non_top1': [], 'noc_top1': [], }

    counter = utils.Counter()

    if args.resume_full_path is not None:
        # load in previous model to continue training
        state, latent_dict, rehearsal_ixs, class_id_to_item_ix_dict, pq = remind.resume(args.streaming_min_class,
                                                                                        args.resume_full_path)

        # validate performance from previous increment
        print('Previous model loaded...computing previous accuracy as sanity check...')
        test_loader = get_data_loader(args.images_dir, args.label_dir, 'val',
                                      args.min_class, args.streaming_min_class,
                                      batch_size=args.batch_size, target=args.datasets)
        _, probas, y_test = remind.predict(test_loader, pq)
        update_accuracies(args, curr_max_class=args.streaming_min_class, remind=remind, pq=pq, accuracies=accuracies)
    else:
        print('\nPerforming base initialization by training on first task...'
              '{}-{}.'.format(args.min_class, args.base_init_classes))

        base_train_loader = get_data_loader(args.images_dir, args.label_dir, 'train',
                                            args.min_class, args.base_init_classes,
                                            batch_size=args.batch_size,
                                            return_item_ix=True, target=args.datasets)

        remind.fit_base_batch(base_train_loader)

        # train pq
        core_model = remind.core_model
        model = ModelWrapper(core_model, output_layer_names=[args.extract_features_from], return_single=True)
        feat_data, label_data, item_ix_data = extract_features(model, base_train_loader,
                                                               len(base_train_loader.dataset),
                                                               num_channels=args.num_channels,
                                                               spatial_feat_dim=args.spatial_feat_dim)

        pq, latent_dict, rehearsal_ixs, class_id_to_item_ix_dict = fit_pq(feat_data, label_data, item_ix_data,
                                                                          args.num_channels,
                                                                          args.spatial_feat_dim, args.num_codebooks,
                                                                          args.codebook_size, counter=counter)

        # save remind model out
        save_full_path = os.path.join(args.save_dir, 'remind_model/')
        remind.save(args.base_init_classes, save_full_path, rehearsal_ixs, latent_dict, class_id_to_item_ix_dict, pq)

        initial_test_loader = get_data_loader(args.images_dir, args.label_dir, 'val', min_class=args.min_class,
                                              max_class=args.base_init_classes, target=args.datasets)
        print('\nComputing base accuracies...')
        base_probas, base_top1, base_top5 = compute_accuracies(initial_test_loader, remind, pq)

        print('\nInitial Test: top1=%0.2f%% -- top5=%0.2f%%' % (base_top1, base_top5))
        utils.save_predictions(base_probas, args.min_class, args.base_init_classes, args.save_dir)
        accuracies['base_classes_top1'].append(float(base_top1))
        accuracies['base_classes_top5'].append(float(base_top5))
        accuracies['seen_classes_top1'].append(float(base_top1))
        accuracies['seen_classes_top5'].append(float(base_top5))

    print('\nBeginning streaming training...')
    for class_ix in range(args.streaming_min_class, args.streaming_max_class, args.class_increment):
        max_class = class_ix + args.class_increment

        print('\nTraining classes {}-{}.'.format(class_ix, max_class))

        train_loader_curr = get_data_loader(args.images_dir, args.label_dir, 'train', class_ix, max_class,
                                            batch_size=args.batch_size,
                                            return_item_ix=True, target=args.datasets)

        # fit model with rehearsal
        remind.fit_incremental_batch(train_loader_curr, latent_dict, pq, rehearsal_ixs=rehearsal_ixs,
                                     class_id_to_item_ix_dict=class_id_to_item_ix_dict,
                                     counter=counter)

        # save remind model out
        save_full_path = os.path.join(args.save_dir, 'remind_model/')
        remind.save(max_class, save_full_path, rehearsal_ixs, latent_dict, class_id_to_item_ix_dict, pq)

        # perform inference
        test_loader = get_data_loader(args.images_dir, args.label_dir, 'val', args.min_class, max_class,
                                      batch_size=args.batch_size, target=args.datasets)
        _, probas, y_test = remind.predict(test_loader, pq)
        update_accuracies(args, curr_max_class=max_class, remind=remind, pq=pq, accuracies=accuracies)

    # final accuracy
    test_loader = get_data_loader(args.images_dir, args.label_dir, 'val', args.min_class, args.streaming_max_class,
                                  batch_size=args.batch_size, target=args.datasets)
    _, probas, y_test = remind.predict(test_loader, pq)
    top1, top5 = utils.accuracy(probas, y_test, topk=(1, 5))
    print('\nFinal: top1=%0.2f%% -- top5=%0.2f%%' % (top1, top5))

    print('\nBeginning fewshot testing...')
    '''deepcopy everything then for to retrain'''
    # for k, p in remind.classifier_F.model.named_parameters():
    #     if 'fc' not in k:
    #         p.requires_grad = False
    remind.lr_scheduler_per_class = None
    remind.use_mixup = False        # no mixup over previous samples
    remind.num_samples = 0      # no replay
    remind.num_epochs = 20

    remind_fixed = copy.deepcopy(remind)
    # latent_dict_fixed = copy.deepcopy(latent_dict)
    # pq_fixed = copy.deepcopy(pq)      can't pickle SwigPyObject objects
    # rehearsal_ixs_fixed = copy.deepcopy(rehearsal_ixs)
    # class_id_to_item_ix_dict_fixed = copy.deepcopy(class_id_to_item_ix_dict)
    # counter_fixed = copy.deepcopy(counter)

    max_class = args.streaming_max_class + 10
    mean_dict = {}
    modes = ['sys', 'pro', 'sub', 'non', 'noc'] if args.datasets == 'CGQA' else ['sys', 'pro', 'non', 'noc']
    for mode in modes:
        remind = copy.deepcopy(remind_fixed)
        # remind, latent_dict, rehearsal_ixs, class_id_to_item_ix_dict, counter = \
        #     copy.deepcopy(remind_fixed), copy.deepcopy(latent_dict_fixed), \
        #     copy.deepcopy(rehearsal_ixs_fixed), \
        #     copy.deepcopy(class_id_to_item_ix_dict_fixed), copy.deepcopy(counter_fixed)
        train_loaders, test_loaders = get_data_loader(
            args.images_dir, args.label_dir, 'train',
            args.streaming_max_class, max_class,
            batch_size=args.batch_size,
            return_item_ix=True, fewshot_mode=mode, target=args.datasets)
        for task_id, (train_loader, test_loader) in enumerate(zip(train_loaders, test_loaders)):

            # fit model with rehearsal
            remind.fit_incremental_batch(train_loader, latent_dict, pq, rehearsal_ixs=rehearsal_ixs,
                                         class_id_to_item_ix_dict=class_id_to_item_ix_dict,
                                         counter=counter, fewshot=True)

            update_accuracies(args, curr_max_class=max_class, remind=remind, pq=pq, accuracies=accuracies,
                              test_loader=test_loader, mode=mode, save=(task_id == len(train_loaders)-1))
        results = np.array(accuracies[f'{mode}_top1'])
        print(f'\n{mode}: {results.mean():.2f} $\\pm$ {1.96 * (results.std() / np.sqrt(len(results))):.2f}')
        mean_dict[mode] = results.mean()

    # 3x1*x2*x3/(x1*x2+x1*x3+x2*x3)
    hm_nov = (3 / (1 / mean_dict['sys'] + 1 / mean_dict['pro'] + 1 / mean_dict['sub'])
              ) if args.datasets == 'CGQA' else (2 / (1 / mean_dict['sys'] + 1 / mean_dict['pro']))

    # 2x1*x2/(x1+x2)
    hm_ref = 2 / (1 / mean_dict['non'] + 1 / mean_dict['noc'])

    hm_all = (5 / (1 / mean_dict['sys'] + 1 / mean_dict['pro'] + 1 / mean_dict['sub'] +
                  1 / mean_dict['non'] + 1 / mean_dict['noc'])
              ) if args.datasets == 'CGQA' else (4 / (1 / mean_dict['sys'] +
                                                      1 / mean_dict['pro'] +
                                                      1 / mean_dict['non'] +
                                                      1 / mean_dict['noc']))

    print(f'hm_nov: {hm_nov:.2f}, hm_ref: {hm_ref:.2f}, hm_all: {hm_all:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # directories and names
    parser.add_argument('--expt_name', type=str)  # name of the experiment
    parser.add_argument('--label_dir', type=str, default=None)  # directory for numpy label files
    parser.add_argument('--images_dir', type=str, default=None)  # directory for ImageNet train/val folders
    parser.add_argument('--save_dir', type=str, required=False)  # directory for saving results
    parser.add_argument('--resume_full_path', type=str, default=None)  # directory of previous model to load
    parser.add_argument('--datasets', type=str, choices=['CGQA', 'COBJ'],
                        default='CGQA')

    # network parameters
    parser.add_argument('--base_arch', type=str, default='ResNet18ClassifyAfterLayer4_1')  # architecture for G
    parser.add_argument('--classifier', type=str, default='ResNet18_StartAt_Layer4_1')  # architecture for F
    parser.add_argument('--classifier_ckpt', type=str, default=None)  # base initialization ckpt
    parser.add_argument('--extract_features_from', type=str,
                        default='model.layer4.0')  # name of the layer to extract features
    parser.add_argument('--num_channels', type=int, default=512)  # number of channels where features are extracted
    parser.add_argument('--spatial_feat_dim', type=int, default=7)  # spatial dimension of features being extracted
    parser.add_argument('--weight_decay', type=float, default=1e-5)  # weight decay for network
    parser.add_argument('--batch_size', type=int, default=100)  # testing batch size

    # pq parameters
    parser.add_argument('--num_codebooks', type=int, default=32)
    parser.add_argument('--codebook_size', type=int, default=256)

    # replay buffer parameters
    parser.add_argument('--rehearsal_samples', type=int, default=50)  # number of replay samples
    parser.add_argument('--max_buffer_size', type=int, default=None)  # maximum number of samples in buffer

    # learning rate parameters
    parser.add_argument('--lr_mode', type=str, choices=['step_lr_per_class', 'cos'],
                        default='step_lr_per_class')  # decay the lr per class
    parser.add_argument('--lr_step_size', type=int, default=100)
    parser.add_argument('--start_lr', type=float, default=0.1)  # starting lr for class
    parser.add_argument('--end_lr', type=float, default=0.001)  # ending lr for class

    # augmentation parameters
    parser.add_argument('--use_random_resized_crops', action='store_true')
    parser.add_argument('--use_mixup', action='store_true')
    parser.add_argument('--mixup_alpha', type=float, default=0.1)

    # streaming setup
    parser.add_argument('--num_epochs', type=int, default=50)  # total number of epochs
    parser.add_argument('--num_classes', type=int, default=110)  # total number of classes  100+10
    parser.add_argument('--min_class', type=int, default=0)  # overall minimum class
    parser.add_argument('--base_init_classes', type=int, default=10)  # number of base init classes
    parser.add_argument('--class_increment', type=int, default=10)  # how often to evaluate
    parser.add_argument('--streaming_min_class', type=int, default=0)  # class to begin stream training
    parser.add_argument('--streaming_max_class', type=int, default=100)  # class to end stream training  100

    # get arguments and print them out and make any necessary directories
    args = parser.parse_args()
    if args.save_dir is None:
        args.save_dir = '../../REMIND-experiments/' + args.expt_name

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # if args.lr_mode == 'step_lr_per_class':
    args.lr_gamma = np.exp(args.lr_step_size * np.log(args.end_lr / args.start_lr) / 1300)

    print("Arguments {}".format(json.dumps(vars(args), indent=4, sort_keys=True)))

    # make model and begin stream training
    remind = REMINDModel(num_classes=args.num_classes, classifier_G=args.base_arch,
                         extract_features_from=args.extract_features_from, classifier_F=args.classifier,
                         classifier_ckpt=args.classifier_ckpt,
                         weight_decay=args.weight_decay, lr_mode=args.lr_mode, lr_step_size=args.lr_step_size,
                         start_lr=args.start_lr, end_lr=args.end_lr, lr_gamma=args.lr_gamma,
                         num_samples=args.rehearsal_samples, use_mixup=args.use_mixup, mixup_alpha=args.mixup_alpha,
                         grad_clip=None, num_channels=args.num_channels, num_feats=args.spatial_feat_dim,
                         num_codebooks=args.num_codebooks, codebook_size=args.codebook_size,
                         use_random_resize_crops=args.use_random_resized_crops,
                         max_buffer_size=args.max_buffer_size,
                         num_epochs=args.num_epochs)
    streaming(args, remind)
