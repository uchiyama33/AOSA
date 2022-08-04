img_norm_cfg = dict(
    mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], to_rgb=False)
crop_size = (320, 768)
sintel_global_transform = dict(
    translates=(0.05, 0.05),
    zoom=(1.0, 1.2),
    shear=(0.95, 1.1),
    rotate=(-5.0, 5.0))
sintel_relative_transform = dict(
    translates=(0.00375, 0.00375),
    zoom=(0.985, 1.015),
    shear=(1.0, 1.0),
    rotate=(-1.0, 1.0))
sintel_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_occ=True),
    dict(
        type='ColorJitter',
        brightness=0.5,
        contrast=0.5,
        saturation=0.5,
        hue=0.5),
    dict(type='RandomGamma', gamma_range=(0.7, 1.5)),
    dict(
        type='Normalize',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        to_rgb=False),
    dict(type='GaussianNoise', sigma_range=(0, 0.04), clamp_range=(0.0, 1.0)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='RandomAffine',
        global_transform=dict(
            translates=(0.05, 0.05),
            zoom=(1.0, 1.2),
            shear=(0.95, 1.1),
            rotate=(-5.0, 5.0)),
        relative_transform=dict(
            translates=(0.00375, 0.00375),
            zoom=(0.985, 1.015),
            shear=(1.0, 1.0),
            rotate=(-1.0, 1.0))),
    dict(type='RandomCrop', crop_size=(320, 768)),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs', 'flow_gt'],
        meta_keys=['img_fields', 'ann_fields', 'filename1', 'filename2',
                   'ori_filename1', 'ori_filename2', 'filename_flow',
                   'ori_filename_flow', 'ori_shape', 'img_shape',
                   'img_norm_cfg'])
]
sintel_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='InputResize', exponent=6),
    dict(
        type='Normalize',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        to_rgb=False),
    dict(type='TestFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs'],
        meta_keys=['flow_gt', 'filename1', 'filename2', 'ori_filename1',
                   'ori_filename2', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'scale_factor', 'pad_shape'])
]
sintel_clean_train = dict(
    type='Sintel',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_occ=True),
        dict(
            type='ColorJitter',
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.5),
        dict(type='RandomGamma', gamma_range=(0.7, 1.5)),
        dict(
            type='Normalize',
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
            to_rgb=False),
        dict(
            type='GaussianNoise',
            sigma_range=(0, 0.04),
            clamp_range=(0.0, 1.0)),
        dict(type='RandomFlip', prob=0.5, direction='horizontal'),
        dict(type='RandomFlip', prob=0.5, direction='vertical'),
        dict(
            type='RandomAffine',
            global_transform=dict(
                translates=(0.05, 0.05),
                zoom=(1.0, 1.2),
                shear=(0.95, 1.1),
                rotate=(-5.0, 5.0)),
            relative_transform=dict(
                translates=(0.00375, 0.00375),
                zoom=(0.985, 1.015),
                shear=(1.0, 1.0),
                rotate=(-1.0, 1.0))),
        dict(type='RandomCrop', crop_size=(320, 768)),
        dict(type='DefaultFormatBundle'),
        dict(
            type='Collect',
            keys=['imgs', 'flow_gt'],
            meta_keys=['img_fields', 'ann_fields', 'filename1', 'filename2',
                       'ori_filename1', 'ori_filename2', 'filename_flow',
                       'ori_filename_flow', 'ori_shape', 'img_shape',
                       'img_norm_cfg'])
    ],
    data_root='data/Sintel',
    test_mode=False,
    pass_style='clean')
sintel_final_train = dict(
    type='Sintel',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_occ=True),
        dict(
            type='ColorJitter',
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.5),
        dict(type='RandomGamma', gamma_range=(0.7, 1.5)),
        dict(
            type='Normalize',
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
            to_rgb=False),
        dict(
            type='GaussianNoise',
            sigma_range=(0, 0.04),
            clamp_range=(0.0, 1.0)),
        dict(type='RandomFlip', prob=0.5, direction='horizontal'),
        dict(type='RandomFlip', prob=0.5, direction='vertical'),
        dict(
            type='RandomAffine',
            global_transform=dict(
                translates=(0.05, 0.05),
                zoom=(1.0, 1.2),
                shear=(0.95, 1.1),
                rotate=(-5.0, 5.0)),
            relative_transform=dict(
                translates=(0.00375, 0.00375),
                zoom=(0.985, 1.015),
                shear=(1.0, 1.0),
                rotate=(-1.0, 1.0))),
        dict(type='RandomCrop', crop_size=(320, 768)),
        dict(type='DefaultFormatBundle'),
        dict(
            type='Collect',
            keys=['imgs', 'flow_gt'],
            meta_keys=['img_fields', 'ann_fields', 'filename1', 'filename2',
                       'ori_filename1', 'ori_filename2', 'filename_flow',
                       'ori_filename_flow', 'ori_shape', 'img_shape',
                       'img_norm_cfg'])
    ],
    data_root='data/Sintel',
    test_mode=False,
    pass_style='final')
sintel_clean_test = dict(
    type='Sintel',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='InputResize', exponent=6),
        dict(
            type='Normalize',
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
            to_rgb=False),
        dict(type='TestFormatBundle'),
        dict(
            type='Collect',
            keys=['imgs'],
            meta_keys=['flow_gt', 'filename1', 'filename2', 'ori_filename1',
                       'ori_filename2', 'ori_shape', 'img_shape',
                       'img_norm_cfg', 'scale_factor', 'pad_shape'])
    ],
    data_root='data/Sintel',
    test_mode=True,
    pass_style='clean')
sintel_final_test = dict(
    type='Sintel',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='InputResize', exponent=6),
        dict(
            type='Normalize',
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
            to_rgb=False),
        dict(type='TestFormatBundle'),
        dict(
            type='Collect',
            keys=['imgs'],
            meta_keys=['flow_gt', 'filename1', 'filename2', 'ori_filename1',
                       'ori_filename2', 'ori_shape', 'img_shape',
                       'img_norm_cfg', 'scale_factor', 'pad_shape'])
    ],
    data_root='data/Sintel',
    test_mode=True,
    pass_style='final')
kitti_global_transform = dict(
    translates=(0.02, 0.02),
    zoom=(0.98, 1.02),
    shear=(1.0, 1.0),
    rotate=(-0.5, 0.5))
kitti_relative_transform = dict(
    translates=(0.0025, 0.0025),
    zoom=(0.99, 1.01),
    shear=(1.0, 1.0),
    rotate=(-0.5, 0.5))
kitti_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', sparse=True),
    dict(
        type='ColorJitter',
        brightness=0.05,
        contrast=0.2,
        saturation=0.25,
        hue=0.1),
    dict(type='RandomGamma', gamma_range=(0.7, 1.5)),
    dict(
        type='Normalize',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        to_rgb=False),
    dict(type='GaussianNoise', sigma_range=(0, 0.02), clamp_range=(0.0, 1.0)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='RandomAffine',
        global_transform=dict(
            translates=(0.02, 0.02),
            zoom=(0.98, 1.02),
            shear=(1.0, 1.0),
            rotate=(-0.5, 0.5)),
        relative_transform=dict(
            translates=(0.0025, 0.0025),
            zoom=(0.99, 1.01),
            shear=(1.0, 1.0),
            rotate=(-0.5, 0.5))),
    dict(type='RandomCrop', crop_size=(320, 768)),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs', 'flow_gt', 'valid'],
        meta_keys=['img_fields', 'ann_fields', 'filename1', 'filename2',
                   'ori_filename1', 'ori_filename2', 'filename_flow',
                   'ori_filename_flow', 'ori_shape', 'img_shape',
                   'img_norm_cfg'])
]
kitti2015_train = dict(
    type='KITTI2015',
    data_root='data/kitti2015',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', sparse=True),
        dict(
            type='ColorJitter',
            brightness=0.05,
            contrast=0.2,
            saturation=0.25,
            hue=0.1),
        dict(type='RandomGamma', gamma_range=(0.7, 1.5)),
        dict(
            type='Normalize',
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
            to_rgb=False),
        dict(
            type='GaussianNoise',
            sigma_range=(0, 0.02),
            clamp_range=(0.0, 1.0)),
        dict(type='RandomFlip', prob=0.5, direction='horizontal'),
        dict(type='RandomFlip', prob=0.5, direction='vertical'),
        dict(
            type='RandomAffine',
            global_transform=dict(
                translates=(0.02, 0.02),
                zoom=(0.98, 1.02),
                shear=(1.0, 1.0),
                rotate=(-0.5, 0.5)),
            relative_transform=dict(
                translates=(0.0025, 0.0025),
                zoom=(0.99, 1.01),
                shear=(1.0, 1.0),
                rotate=(-0.5, 0.5))),
        dict(type='RandomCrop', crop_size=(320, 768)),
        dict(type='DefaultFormatBundle'),
        dict(
            type='Collect',
            keys=['imgs', 'flow_gt', 'valid'],
            meta_keys=['img_fields', 'ann_fields', 'filename1', 'filename2',
                       'ori_filename1', 'ori_filename2', 'filename_flow',
                       'ori_filename_flow', 'ori_shape', 'img_shape',
                       'img_norm_cfg'])
    ],
    test_mode=False)
hd1k_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', sparse=True),
    dict(type='RandomCrop', crop_size=(436, 1024)),
    dict(
        type='ColorJitter',
        brightness=0.05,
        contrast=0.2,
        saturation=0.25,
        hue=0.1),
    dict(type='RandomGamma', gamma_range=(0.7, 1.5)),
    dict(
        type='Normalize',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        to_rgb=False),
    dict(type='GaussianNoise', sigma_range=(0, 0.02), clamp_range=(0.0, 1.0)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='RandomAffine',
        global_transform=dict(
            translates=(0.02, 0.02),
            zoom=(0.98, 1.02),
            shear=(1.0, 1.0),
            rotate=(-0.5, 0.5)),
        relative_transform=dict(
            translates=(0.0025, 0.0025),
            zoom=(0.99, 1.01),
            shear=(1.0, 1.0),
            rotate=(-0.5, 0.5))),
    dict(type='RandomCrop', crop_size=(320, 768)),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs', 'flow_gt', 'valid'],
        meta_keys=['img_fields', 'ann_fields', 'filename1', 'filename2',
                   'ori_filename1', 'ori_filename2', 'filename_flow',
                   'ori_filename_flow', 'ori_shape', 'img_shape',
                   'img_norm_cfg'])
]
hd1k_train = ({
    'type':
    'HD1K',
    'data_root':
    'data/hd1k',
    'pipeline': [{
        'type': 'LoadImageFromFile'
    }, {
        'type': 'LoadAnnotations',
        'sparse': True
    }, {
        'type': 'RandomCrop',
        'crop_size': (436, 1024)
    }, {
        'type': 'ColorJitter',
        'brightness': 0.05,
        'contrast': 0.2,
        'saturation': 0.25,
        'hue': 0.1
    }, {
        'type': 'RandomGamma',
        'gamma_range': (0.7, 1.5)
    }, {
        'type': 'Normalize',
        'mean': [0.0, 0.0, 0.0],
        'std': [255.0, 255.0, 255.0],
        'to_rgb': False
    }, {
        'type': 'GaussianNoise',
        'sigma_range': (0, 0.02),
        'clamp_range': (0.0, 1.0)
    }, {
        'type': 'RandomFlip',
        'prob': 0.5,
        'direction': 'horizontal'
    }, {
        'type': 'RandomFlip',
        'prob': 0.5,
        'direction': 'vertical'
    }, {
        'type': 'RandomAffine',
        'global_transform': {
            'translates': (0.02, 0.02),
            'zoom': (0.98, 1.02),
            'shear': (1.0, 1.0),
            'rotate': (-0.5, 0.5)
        },
        'relative_transform': {
            'translates': (0.0025, 0.0025),
            'zoom': (0.99, 1.01),
            'shear': (1.0, 1.0),
            'rotate': (-0.5, 0.5)
        }
    }, {
        'type': 'RandomCrop',
        'crop_size': (320, 768)
    }, {
        'type': 'DefaultFormatBundle'
    }, {
        'type':
        'Collect',
        'keys': ['imgs', 'flow_gt', 'valid'],
        'meta_keys':
        ['img_fields', 'ann_fields', 'filename1', 'filename2', 'ori_filename1',
         'ori_filename2', 'filename_flow', 'ori_filename_flow', 'ori_shape',
         'img_shape', 'img_norm_cfg']
    }],
    'test_mode':
    False
}, )
data = dict(
    train_dataloader=dict(
        samples_per_gpu=1,
        workers_per_gpu=2,
        drop_last=True,
        sample_ratio=(0.5, 0.25, 0.25),
        persistent_workers=True),
    val_dataloader=dict(samples_per_gpu=1, workers_per_gpu=5, shuffle=False),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=5, shuffle=False),
    train=[[{
        'type':
        'Sintel',
        'pipeline': [{
            'type': 'LoadImageFromFile'
        }, {
            'type': 'LoadAnnotations',
            'with_occ': True
        }, {
            'type': 'ColorJitter',
            'brightness': 0.5,
            'contrast': 0.5,
            'saturation': 0.5,
            'hue': 0.5
        }, {
            'type': 'RandomGamma',
            'gamma_range': (0.7, 1.5)
        }, {
            'type': 'Normalize',
            'mean': [0.0, 0.0, 0.0],
            'std': [255.0, 255.0, 255.0],
            'to_rgb': False
        }, {
            'type': 'GaussianNoise',
            'sigma_range': (0, 0.04),
            'clamp_range': (0.0, 1.0)
        }, {
            'type': 'RandomFlip',
            'prob': 0.5,
            'direction': 'horizontal'
        }, {
            'type': 'RandomFlip',
            'prob': 0.5,
            'direction': 'vertical'
        }, {
            'type': 'RandomAffine',
            'global_transform': {
                'translates': (0.05, 0.05),
                'zoom': (1.0, 1.2),
                'shear': (0.95, 1.1),
                'rotate': (-5.0, 5.0)
            },
            'relative_transform': {
                'translates': (0.00375, 0.00375),
                'zoom': (0.985, 1.015),
                'shear': (1.0, 1.0),
                'rotate': (-1.0, 1.0)
            }
        }, {
            'type': 'RandomCrop',
            'crop_size': (320, 768)
        }, {
            'type': 'DefaultFormatBundle'
        }, {
            'type':
            'Collect',
            'keys': ['imgs', 'flow_gt'],
            'meta_keys':
            ['img_fields', 'ann_fields', 'filename1', 'filename2',
             'ori_filename1', 'ori_filename2', 'filename_flow',
             'ori_filename_flow', 'ori_shape', 'img_shape', 'img_norm_cfg']
        }],
        'data_root':
        'data/Sintel',
        'test_mode':
        False,
        'pass_style':
        'clean'
    }, {
        'type':
        'Sintel',
        'pipeline': [{
            'type': 'LoadImageFromFile'
        }, {
            'type': 'LoadAnnotations',
            'with_occ': True
        }, {
            'type': 'ColorJitter',
            'brightness': 0.5,
            'contrast': 0.5,
            'saturation': 0.5,
            'hue': 0.5
        }, {
            'type': 'RandomGamma',
            'gamma_range': (0.7, 1.5)
        }, {
            'type': 'Normalize',
            'mean': [0.0, 0.0, 0.0],
            'std': [255.0, 255.0, 255.0],
            'to_rgb': False
        }, {
            'type': 'GaussianNoise',
            'sigma_range': (0, 0.04),
            'clamp_range': (0.0, 1.0)
        }, {
            'type': 'RandomFlip',
            'prob': 0.5,
            'direction': 'horizontal'
        }, {
            'type': 'RandomFlip',
            'prob': 0.5,
            'direction': 'vertical'
        }, {
            'type': 'RandomAffine',
            'global_transform': {
                'translates': (0.05, 0.05),
                'zoom': (1.0, 1.2),
                'shear': (0.95, 1.1),
                'rotate': (-5.0, 5.0)
            },
            'relative_transform': {
                'translates': (0.00375, 0.00375),
                'zoom': (0.985, 1.015),
                'shear': (1.0, 1.0),
                'rotate': (-1.0, 1.0)
            }
        }, {
            'type': 'RandomCrop',
            'crop_size': (320, 768)
        }, {
            'type': 'DefaultFormatBundle'
        }, {
            'type':
            'Collect',
            'keys': ['imgs', 'flow_gt'],
            'meta_keys':
            ['img_fields', 'ann_fields', 'filename1', 'filename2',
             'ori_filename1', 'ori_filename2', 'filename_flow',
             'ori_filename_flow', 'ori_shape', 'img_shape', 'img_norm_cfg']
        }],
        'data_root':
        'data/Sintel',
        'test_mode':
        False,
        'pass_style':
        'final'
    }], {
        'type':
        'KITTI2015',
        'data_root':
        'data/kitti2015',
        'pipeline': [{
            'type': 'LoadImageFromFile'
        }, {
            'type': 'LoadAnnotations',
            'sparse': True
        }, {
            'type': 'ColorJitter',
            'brightness': 0.05,
            'contrast': 0.2,
            'saturation': 0.25,
            'hue': 0.1
        }, {
            'type': 'RandomGamma',
            'gamma_range': (0.7, 1.5)
        }, {
            'type': 'Normalize',
            'mean': [0.0, 0.0, 0.0],
            'std': [255.0, 255.0, 255.0],
            'to_rgb': False
        }, {
            'type': 'GaussianNoise',
            'sigma_range': (0, 0.02),
            'clamp_range': (0.0, 1.0)
        }, {
            'type': 'RandomFlip',
            'prob': 0.5,
            'direction': 'horizontal'
        }, {
            'type': 'RandomFlip',
            'prob': 0.5,
            'direction': 'vertical'
        }, {
            'type': 'RandomAffine',
            'global_transform': {
                'translates': (0.02, 0.02),
                'zoom': (0.98, 1.02),
                'shear': (1.0, 1.0),
                'rotate': (-0.5, 0.5)
            },
            'relative_transform': {
                'translates': (0.0025, 0.0025),
                'zoom': (0.99, 1.01),
                'shear': (1.0, 1.0),
                'rotate': (-0.5, 0.5)
            }
        }, {
            'type': 'RandomCrop',
            'crop_size': (320, 768)
        }, {
            'type': 'DefaultFormatBundle'
        }, {
            'type':
            'Collect',
            'keys': ['imgs', 'flow_gt', 'valid'],
            'meta_keys':
            ['img_fields', 'ann_fields', 'filename1', 'filename2',
             'ori_filename1', 'ori_filename2', 'filename_flow',
             'ori_filename_flow', 'ori_shape', 'img_shape', 'img_norm_cfg']
        }],
        'test_mode':
        False
    },
           ({
               'type':
               'HD1K',
               'data_root':
               'data/hd1k',
               'pipeline': [{
                   'type': 'LoadImageFromFile'
               }, {
                   'type': 'LoadAnnotations',
                   'sparse': True
               }, {
                   'type': 'RandomCrop',
                   'crop_size': (436, 1024)
               }, {
                   'type': 'ColorJitter',
                   'brightness': 0.05,
                   'contrast': 0.2,
                   'saturation': 0.25,
                   'hue': 0.1
               }, {
                   'type': 'RandomGamma',
                   'gamma_range': (0.7, 1.5)
               }, {
                   'type': 'Normalize',
                   'mean': [0.0, 0.0, 0.0],
                   'std': [255.0, 255.0, 255.0],
                   'to_rgb': False
               }, {
                   'type': 'GaussianNoise',
                   'sigma_range': (0, 0.02),
                   'clamp_range': (0.0, 1.0)
               }, {
                   'type': 'RandomFlip',
                   'prob': 0.5,
                   'direction': 'horizontal'
               }, {
                   'type': 'RandomFlip',
                   'prob': 0.5,
                   'direction': 'vertical'
               }, {
                   'type': 'RandomAffine',
                   'global_transform': {
                       'translates': (0.02, 0.02),
                       'zoom': (0.98, 1.02),
                       'shear': (1.0, 1.0),
                       'rotate': (-0.5, 0.5)
                   },
                   'relative_transform': {
                       'translates': (0.0025, 0.0025),
                       'zoom': (0.99, 1.01),
                       'shear': (1.0, 1.0),
                       'rotate': (-0.5, 0.5)
                   }
               }, {
                   'type': 'RandomCrop',
                   'crop_size': (320, 768)
               }, {
                   'type': 'DefaultFormatBundle'
               }, {
                   'type':
                   'Collect',
                   'keys': ['imgs', 'flow_gt', 'valid'],
                   'meta_keys': ['img_fields', 'ann_fields', 'filename1',
                                 'filename2', 'ori_filename1', 'ori_filename2',
                                 'filename_flow', 'ori_filename_flow',
                                 'ori_shape', 'img_shape', 'img_norm_cfg']
               }],
               'test_mode':
               False
           }, )],
    val=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='Sintel',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(type='InputResize', exponent=6),
                    dict(
                        type='Normalize',
                        mean=[0.0, 0.0, 0.0],
                        std=[255.0, 255.0, 255.0],
                        to_rgb=False),
                    dict(type='TestFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['imgs'],
                        meta_keys=['flow_gt', 'filename1', 'filename2',
                                   'ori_filename1', 'ori_filename2',
                                   'ori_shape', 'img_shape', 'img_norm_cfg',
                                   'scale_factor', 'pad_shape'])
                ],
                data_root='data/Sintel',
                test_mode=True,
                pass_style='clean'),
            dict(
                type='Sintel',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(type='InputResize', exponent=6),
                    dict(
                        type='Normalize',
                        mean=[0.0, 0.0, 0.0],
                        std=[255.0, 255.0, 255.0],
                        to_rgb=False),
                    dict(type='TestFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['imgs'],
                        meta_keys=['flow_gt', 'filename1', 'filename2',
                                   'ori_filename1', 'ori_filename2',
                                   'ori_shape', 'img_shape', 'img_norm_cfg',
                                   'scale_factor', 'pad_shape'])
                ],
                data_root='data/Sintel',
                test_mode=True,
                pass_style='final')
        ],
        separate_eval=True),
    test=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='Sintel',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(type='InputResize', exponent=6),
                    dict(
                        type='Normalize',
                        mean=[0.0, 0.0, 0.0],
                        std=[255.0, 255.0, 255.0],
                        to_rgb=False),
                    dict(type='TestFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['imgs'],
                        meta_keys=['flow_gt', 'filename1', 'filename2',
                                   'ori_filename1', 'ori_filename2',
                                   'ori_shape', 'img_shape', 'img_norm_cfg',
                                   'scale_factor', 'pad_shape'])
                ],
                data_root='data/Sintel',
                test_mode=True,
                pass_style='clean'),
            dict(
                type='Sintel',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(type='InputResize', exponent=6),
                    dict(
                        type='Normalize',
                        mean=[0.0, 0.0, 0.0],
                        std=[255.0, 255.0, 255.0],
                        to_rgb=False),
                    dict(type='TestFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['imgs'],
                        meta_keys=['flow_gt', 'filename1', 'filename2',
                                   'ori_filename1', 'ori_filename2',
                                   'ori_shape', 'img_shape', 'img_norm_cfg',
                                   'scale_factor', 'pad_shape'])
                ],
                data_root='data/Sintel',
                test_mode=True,
                pass_style='final')
        ],
        separate_eval=True))
optimizer = dict(
    type='Adam', lr=5e-05, weight_decay=0.0004, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='MultiStage',
    by_epoch=False,
    gammas=[0.5, 0.5, 0.5, 0.5, 0.5],
    milestone_lrs=[5e-05, 3e-05, 2e-05, 1e-05, 5e-06],
    milestone_iters=[0, 150000, 300000, 450000, 600000],
    steps=[[
        45000, 65000, 85000, 95000, 97500, 100000, 110000, 120000, 130000,
        140000
    ],
           [
               195000, 215000, 235000, 245000, 247500, 250000, 260000, 270000,
               280000, 290000
           ],
           [
               345000, 365000, 385000, 395000, 397500, 400000, 410000, 420000,
               430000, 440000
           ],
           [
               495000, 515000, 535000, 545000, 547500, 550000, 560000, 570000,
               580000, 590000
           ],
           [
               645000, 665000, 685000, 695000, 697500, 700000, 710000, 720000,
               730000, 740000
           ]])
runner = dict(type='IterBasedRunner', max_iters=750000)
checkpoint_config = dict(by_epoch=False, interval=50000)
evaluation = dict(interval=50000, metric='EPE')
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'work_dir/upload_ready/pwcnet/pwcnet_8x1_sfine_flyingthings3d_subset_384x768.pth'
resume_from = None
workflow = [('train', 1)]
model = dict(
    type='PWCNet',
    encoder=dict(
        type='PWCNetEncoder',
        in_channels=3,
        net_type='Basic',
        pyramid_levels=[
            'level1', 'level2', 'level3', 'level4', 'level5', 'level6'
        ],
        out_channels=(16, 32, 64, 96, 128, 196),
        strides=(2, 2, 2, 2, 2, 2),
        dilations=(1, 1, 1, 1, 1, 1),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
    decoder=dict(
        type='PWCNetDecoder',
        in_channels=dict(
            level6=81, level5=213, level4=181, level3=149, level2=117),
        flow_div=20.0,
        corr_cfg=dict(type='Correlation', max_displacement=4, padding=0),
        warp_cfg=dict(type='Warp', align_corners=True, use_mask=True),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        scaled=False,
        post_processor=dict(type='ContextNet', in_channels=565),
        flow_loss=dict(
            type='MultiLevelEPE',
            p=1,
            q=0.4,
            eps=0.01,
            reduction='sum',
            resize_flow='upsample',
            weights=dict(
                level2=0.005,
                level3=0.01,
                level4=0.02,
                level5=0.08,
                level6=0.32))),
    train_cfg=dict(),
    test_cfg=dict(),
    init_cfg=dict(
        type='Kaiming',
        nonlinearity='leaky_relu',
        layer=['Conv2d', 'ConvTranspose2d'],
        mode='fan_in',
        bias=0))
work_dir = 'work_dir/pwc+'
gpu_ids = range(0, 1)
