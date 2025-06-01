import streamlit as st
import pandas as pd
import numpy as np
import yaml
import cv2
import os
import logging
import traceback
from image_segmentation.segment_inference import face_segment
from runners.image_editing import Diffusion
from image_landmark_transform.face_landmark import face_landmark_transform
from image_artifact_fill.artifact_fill import face_artifact_fill
from inference import resize_image, dict2namespace

class app:
    def __init__(self):
        self.args = self.create_args()
        self.init_app()
        self.config = self.create_config()
        run = st.button('RUN')
        if self.args['target_image'] and self.args['source_image'] and run:
            self.pipeline()
        
    def init_app(self):
        st.title('Realistic Hairstyle Try-On')
        st.subheader('Input Images')
        self.args['target_image'] = st.file_uploader(
            'Target image (The person whose FACE you desire)',
            type=['png', 'jpg', 'jpeg']
        )
        self.args['source_image'] = st.file_uploader(
            'Source image (The person whose HAIR you desire)',
            type=['png', 'jpg', 'jpeg']
        )

        if self.args['target_image'] and self.args['source_image']:
            self.target_image = self.read_image_from_streamlit(self.args['target_image'])
            self.source_image = self.read_image_from_streamlit(self.args['source_image'])
            images = [self.target_image, self.source_image]
            captions = ['Target image', 'Source image']
            st.image(images, width=300, caption=captions)

        st.sidebar.header('Input Some Parameters (Defaults Are Fine)')
        st.sidebar.subheader('SDEdit Parameters')

        self.args['seed'] = st.sidebar.number_input(
            'Random seed', min_value=0, value=1234, step=1, format='%d'
        )
        self.args['sample_step'] = st.sidebar.number_input(
            'Number of generated images', min_value=1, max_value=5, value=1, step=1, format='%d'
        )
        self.args['t'] = st.sidebar.number_input(
            'Noise scale (higher = slower, lower = less realistic)',
            min_value=0, max_value=2000, value=500, step=1, format='%d'
        )
        self.args['erode_kernel_size'] = st.sidebar.number_input(
            'Erode kernel size', min_value=0, max_value=10, value=7, step=1, format='%d'
        )

    def create_args(self):
        args = {}
        args['seg_model_path'] = os.path.join(
            'image_segmentation', 'face_segment_checkpoints_256.pth.tar'
        )
        args['image_size'] = (256, 256)
        args['input_image_size'] = (256, 256)
        args['label_config'] = os.path.join(
            'image_segmentation', 'label.yml'
        )
        # SDEdit defaults
        args['exp'] = 'exp'
        args['verbose'] = 'info'
        args['sample'] = True
        args['image_folder'] = 'images'
        args['ni'] = True
        args['is_erode_mask'] = True
        return args

    @st.cache_data
    def create_config(_self, config_file_path=os.path.join('configs', 'celeba.yml')):
        # Parse YAML config file
        with open(config_file_path, 'r') as f:
            config = yaml.safe_load(f)
        return dict2namespace(config)

    def read_image_from_streamlit(self, uploaded_file):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def pipeline(self):
        handler = face_segment(
            seg_model_path=self.args['seg_model_path'],
            label_config=self.args['label_config'],
            input_image_size=self.args['input_image_size']
        )

        # Segmentation
        target_mask = handler.segmenting(image=self.target_image)
        source_mask = handler.segmenting(image=self.source_image)

        # Resize
        target_image = resize_image(self.target_image, self.args['image_size'])
        source_image = resize_image(self.source_image, self.args['image_size'])
        target_mask = resize_image(target_mask, self.args['image_size'])
        source_mask = resize_image(source_mask, self.args['image_size'])

        # Landmark transform
        outputs = face_landmark_transform(
            target_image, target_mask, source_image, source_mask
        )
        transformed_image = outputs['result_image']
        transformed_mask = outputs['result_mask']
        transformed_segment = handler.segmenting(image=transformed_image)

        # Artifact fill
        filled_image = face_artifact_fill(
            target_image, target_mask,
            transformed_image, transformed_mask, transformed_segment
        )

        st.image(
            [target_mask, source_mask, transformed_image, filled_image],
            width=300,
            caption=['Target mask', 'Source mask', 'Transformed image', 'Filled image'],
            clamp=True
        )

        # SDEdit mask processing
        sde_mask = outputs['only_fixed_face']
        if self.args['is_erode_mask']:
            k = self.args['erode_kernel_size']
            kernel = np.ones((k, k), np.uint8)
            sde_mask = cv2.erode(sde_mask, kernel, iterations=1)

        try:
            runner = Diffusion(
                image_folder=self.args['image_folder'],
                sample_step=self.args['sample_step'],
                total_noise_levels=self.args['t'],
                config=self.config
            )
            self.show_images = runner.image_editing_sample_for_streamlit(
                filled_image, sde_mask
            )
            imgs = list(self.show_images.values())
            caps = list(self.show_images.keys())
            st.image(imgs, width=100, caption=caps, clamp=True)
            for i in range(self.args['sample_step']):
                st.image(
                    self.show_images[f'samples_{i}'],
                    width=300,
                    caption=f'Final image {i+1}',
                    clamp=True
                )
        except Exception as e:
            logging.error(traceback.format_exc())

if __name__ == '__main__':
    app()
