
import os
import shutil
import evaluation_settings as s
from tf_unet import unet
import glob

def run(sequence_dir, working_dir, force=False):
    multitime = os.path.basename(sequence_dir).split('_', maxsplit=1)[1]
    # Get appropriate model
    model_dir = glob.glob(os.path.join(working_dir, s.stages[4], '*' + multitime + '*'))[0]
    # Create output dir
    output_dir = os.path.join(working_dir, s.stages[6], os.path.basename(model_dir) + os.path.basename(sequence_dir))
    # only retest if necessary
    if not os.path.exists(output_dir) or force:
        # delete existing
        if force and os.path.exists(output_dir):
            try:
                shutil.rmtree(output_dir)
            except Exception as e:
                print(e)

        # make directory
        os.makedirs(output_dir)

        net = unet.Unet(
            channels=s.network['channels'],
            n_class=s.network['classes'],
            layers=s.network['layers'],
            features_root=s.network['features_root'],
            cost_kwargs=dict(class_weights=s.network['class_weights'])
        )
        # Run prediction
        net.predict_no_label(
            model_path=os.path.join(model_dir, 'model.cpkt'),
            images_dir=sequence_dir, output_dir=output_dir,)
    else:
        print(os.path.basename(output_dir), ' already exists. Skipping.')
