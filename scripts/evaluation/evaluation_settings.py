stages = [
    '1_fetch_videos',
    '2_extract_frames',
    '3_select_for_labelling',
    '4_select_training_subsets',
    '5_train',
    '6_test',
    '7_predict',
    '8_compute_index',
    '9_classify',
    '10_compare',
    '11_analyze'
]

frame_extraction_combinations_large = [
    [0, 0, 0],
    # [0, 0, 0.1],
    # [0, 0, 0.2],
    # [0, 0, 0.5],
    # [0, 0, 1],
    # [0, 0, 5],
    # [0, 0.1, 0.2],
    # [0, 0.2, 0.4],
    # [0, 0.5, 1],
    # [0, 1, 2],
    # [0, 5, 10],
    # [0, 0.1, 0.3],
    # [0, 0.2, 0.6],
    # [0, 0.5, 1.5],
    # [0, 0.2, 1.2]
]

frame_extraction_combinations_small = [
    [0, 0, 0]
    #[0, 0, 5],
    #[0, 0, 0.2],
    #[0, 0.2, 0.4]
]

frame_extraction_new_dim = (640, 360)
select_sample_images = {
    # "intra-event": {
    #     "cameras": ["cam1"],
    #     "name": 'intra',
    #     "start": "161007_110800",
    #     "end": "161007_121000",
    #     "count": 100,
    #     "encourage_flooded": 1  # weight given to flooded frames to encourage selection. the weight given to
    #                             # non-flooded frames is 1
    # },
    # "inter-event": {
    #     "cameras": ["cam1"],
    #     "name": 'inter',
    #     "start": "161007_121000",
    #     "end": "161007_134830",
    #     "count": 30,  # frames to label manually for train-test
    #     "encourage_flooded": 3  # weight to encourage extraction of frames during flooding
    # },
    # "extra": {
    #     "cameras": ["cam2", "cam3", "cam4", "cam5"],
    #     "name": 'extra',
    #     "start": "161007_110800",
    #     "end": "161007_134830",
    #     "count": 30,
    #     "encourage_flooded": 3
    # },
    # "extra2": {
    #     "cameras": ["gopro1"],
    #     "name": 'extra',
    #     "start": "161006_165900",
    #     "end": "161006_174600",
    #     "count": 30,
    #     "encourage_flooded": 3
    # },
    # "intra-event-2": {
    #     "cameras": ["cam5"],
    #     "name": 'intra',
    #     "start": "161007_124600",
    #     "end": "161007_130422",
    #     "count": 50,
    #     "encourage_flooded": 1
    # },
    # "inter-event-2": {
    #     "cameras": ["cam5"],
    #     "name": 'inter',
    #     "start": "161007_131344",
    #     "end": "161007_135600",
    #     "count": 30,
    #     "encourage_flooded": 1
    # },
    "new": {
        "cameras": ["cam1"],
        "name": 'new',
        "start": "161007_122500",
        "end": "161007_123400",
        "count": 0,
        "encourage_flooded": 1  # weight given to flooded frames to encourage selection. the weight given to
    }

}
datasets = {
    "intra": {
        "frac_train": 0.7,
        "frac_validate": 0,
        "frac_test": 0.3
    },
    "inter": {
        "frac_train": 0,
        "frac_validate": 0,
        "frac_test": 1
    },
    "extra": {
        "frac_train": 0,
        "frac_validate": 0,
        "frac_test": 1
    },
    "supervisely": {
        "frac_train": 0.9,
        "frac_validate": 0,
        "frac_test": 0.1
    }
}

network = {
    "channels": 3,
    "classes": 8,  # before: 3
    "layers": 5,
    "features_root": 16,
    "class_weights": [1, 2, 2, 2, 2, 2, 2, 2]  # before: [1, 1, 2]
}

train = {
    "optimizer": "adam",
    "batch_size": 8,
    "training_iters": 100,
    "epochs": 30,
    "display_step": 50,
    "dropout": 0.8
}

prediction_combinations = [
    # {
    #     "data": "cam1_161007A_0_0.2_0.4",
    #     "model": "cam1_intra_0_0.2_0.4__ly4ftr16w2__"
    # },
    # {
    #     "data": "cam1_161006A_0_0.2_0.4",
    #     "model": "cam1cam5_intra_0_0.2_0.4__ly4ftr16w2__"
    # },
    # {
    #      "data": "cam1_161006B_0_0.2_0.4",
    #      "model": "cam1cam5_intra_0_0.2_0.4__ly4ftr16w2__"
    # },
    # {
    #      "data": "cam5_161006B_0_0.2_0.4",
    #      "model": "cam1cam5_intra_0_0.2_0.4__ly4ftr16w2__"
    # },
    # {
    #      "data": "cam4_161006A_0_0.2_0.4",
    #      "model": "cam1cam5_intra_0_0.2_0.4__ly4ftr16w2__"
    # },
    # {
    #      "data": "cam4_161006B_0_0.2_0.4",
    #      "model": "cam1cam5_intra_0_0.2_0.4__ly4ftr16w2__"
    # }
    # {
    #      "data": "cam2_161007A_0_0.2_0.4",
    #      "model": "cam1cam5_intra_0_0.2_0.4__ly4ftr16w2__"
    # },
    # {
    #      "data": "cam3_161007A_0_0.2_0.4",
    #      "model": "cam1cam5_intra_0_0.2_0.4__ly4ftr16w2__"
    # },
    # {
    #      "data": "cam3_161006A_0_0.2_0.4",
    #      "model": "cam1cam5_intra_0_0.2_0.4__ly4ftr16w2__"
    # },
    # {
    #      "data": "cam3_161006B_0_0.2_0.4",
    #      "model": "cam1cam5_intra_0_0.2_0.4__ly4ftr16w2__"
    # },
    # {
    #      "data": "cam2_161006A_0_0.2_0.4",
    #      "model": "cam1cam5_intra_0_0.2_0.4__ly4ftr16w2__"
    # },
    # {
    #      "data": "cam2_161006B_0_0.2_0.4",
    #      "model": "cam1cam5_intra_0_0.2_0.4__ly4ftr16w2__"
    # }
    # from matthew already done
    # {
    #     "data": "cam2_0_0.2_0.4",
    #     "model": "cam1_intra_0_0.2_0.4__ly4ftr16w2__"
    # },
    # {
    #     "data": "cam3_0_0.2_0.4",
    #     "model": "cam1_intra_0_0.2_0.4__ly4ftr16w2__"
    # },
    # {
    #     "data": "cam4_0_0.2_0.4",
    #     "model": "cam1_intra_0_0.2_0.4__ly4ftr16w2__"
    # },
    # {
    #     "data": "cam5_0_0.2_0.4",
    #     "model": "cam1_intra_0_0.2_0.4__ly4ftr16w2__"
    # },
    # {
    #     "data": "gopro1_0_0.2_0.4",
    #     "model": "cam1_intra_0_0.2_0.4__ly4ftr16w2__"
    # },
    # {
    #     "data": "cam1_0_0.2_0.4",
    #     "model": "cam1cam5_intra_0_0.2_0.4__ly4ftr16w2__"
    # },
    # {
    #     "data": "cam5_0_0.2_0.4",
    #     "model": "cam1cam5_intra_0_0.2_0.4__ly4ftr16w2__"
    # },
    # {
    #     "data": "gopro1_0_0.2_0.4",
    #     "model": "cam1cam5_intra_0_0.2_0.4__ly4ftr16w2__"
    # },
    # {
    #      "data": "cam1_161006A_0_0_0",
    #      "model": "all_flipped2_supervisely__ly4ftr16w2__"
    # },
    # {
    #      "data": "cam2_161007A_0_0_0",
    #      "model": "all_flipped2_supervisely__ly4ftr16w2__"
    # },
    {
         "data": "cam4_180531_0_0_0",
         "model": "all_flipped2_supervisely__ly4ftr16w2__"
    },
]
y_scaling = 200
rois = {
    "cam1": {
        "name": "dam_water_level_cam1",
        "top": 0,
        "left": 255,
        "height": 125,
        "width": 323
    },
    "cam5": {
        "name": "basement_cam5",
        "top": 94,
        "left": 211,
        "height": 131,
        "width": 120
    },
    "gopro1": {
        "name": "dam_water_level_gopro1",
        "top": 118,
        "left": 409,
        "height": 142,
        "width": 120
    },
    "Flood": {
        "name": "supervisely_pics",
        "top": 0,
        "left": 0,
        "height": 360,
        "width": 640,
    },
    "cam2": {
        "name": "cam2",
        "top": 60,
        "left": 30,
        "height": 220,
        "width": 500
    },
    "cam4": {
        "name": "dam_water_level_cam4",
        "top": 20,
        "left": 0,
        "height": 330,
        "width": 640
    },
}