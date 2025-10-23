import os
import cv2
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms

from tqdm import tqdm
from torchvision.io.image import decode_image
from scipy.spatial.distance import cosine

from extractor import extractor_factory
from localizer import localizer_factory

class InstanceSearch:
    def __init__(self, 
                 gallery_path: str, 
                 query_path: str, 
                 txt_path: str, 
                 localizer: str = 'fasterrcnn',
                 extractor: str = 'resnet50', 
                 log_file: str = 'rankList.txt',
                 gallery_feature_file: str = "gallery_features.pkl", 
                 query_feature_file: str = "query_features.pkl",                  
                 batch_size: int = 32, ):
        # working directory
        method = f"{localizer}_{extractor}"

        if not os.path.exists(method):
            os.makedirs(method)

        self.gallery_path = gallery_path
        self.query_path = query_path
        self.txt_path = txt_path
        self.batch_size = batch_size
        self.gallery_feature_file = os.path.join(method, gallery_feature_file)
        self.query_feature_file = os.path.join(method, query_feature_file)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # set device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print('[device: gpu]')
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print('[device: mps]')
        else:
            self.device = torch.device("cpu")
            print('[device: cpu]')

        # load localizer
        localizer, preprocess = self._load_localizer(localizer)
        self.localizer = localizer
        self.preprocess = preprocess

        # load extractor
        self.extractor = self._load_extractor(extractor)

        query_bboxes, gallery_bboxes = self._load_bounding_boxes()

        self.query_bboxes = query_bboxes
        self.gallery_bboxes = gallery_bboxes

        # log
        logging.basicConfig(filename=os.path.join(method, log_file), 
                            level=logging.INFO, 
                            filemode='w')
        
        # precomputed and save gallery/query features
        self.gallery_features = self._load_or_extract_gallery_features()
        self.query_features = self._load_or_extract_query_features()
    
    def _load_localizer(self, localizer: str):
        """ load ROI localizer """
        model, transforms = localizer_factory(localizer)
        model = model.to(self.device)
        model.eval()
        return model, transforms
    
    def _load_extractor(self, extractor: str):
        """ load feature extractor """
        model = extractor_factory(extractor)
        model = model.to(self.device)
        model.eval()
        return model
    
    def _load_bounding_boxes(self, threshold: float = 0.6):
        """ retrieve bounding boxes (x, y, w, h) for all query images """
        query_bboxes, gallery_bboxes = {}, {}
        # load annoations
        txt_names = [f for f in os.listdir(self.txt_path) if f.endswith('.txt')]
        for txt_name in txt_names:
            query_img_name = txt_name.replace('.txt', '.jpg')
            txt_file = os.path.join(self.txt_path, txt_name)
            bboxes = self._parse_bbox(txt_file)
            assert len(bboxes) > 0, f'fail to load bounding box {txt_file}!'
            query_bboxes[query_img_name] = bboxes

        # gallery localization
        img_names = [f for f in os.listdir(self.gallery_path) if f.endswith('.jpg')]
        gallery_resize = transforms.Resize((746, 1000))
        for i in tqdm(range(0, len(img_names), self.batch_size), desc="propose gallery ROIs"):
            batch_names = img_names[i:i+self.batch_size]
            batch_images = []
            valid_names = []
            
            # batch images
            for img_name in batch_names:
                img = decode_image(os.path.join(self.gallery_path, img_name))
                # img = cv2.imread(os.path.join(self.gallery_path, img_name))
                if img is not None:
                    img = self.preprocess(img)
                    batch_images.append(gallery_resize(img))
                    valid_names.append(img_name)
            
            if not batch_images:
                continue
                
            # localize ROI
            batch_tensor = torch.stack(batch_images).to(self.device)

            with torch.no_grad():
                batch_predictions = self.localizer(batch_tensor)

            for i, prediction in enumerate(batch_predictions):
                if len(prediction['boxes']) > 0:
                    scores = prediction['scores'].cpu().numpy()
                    boxes = prediction['boxes'].cpu().numpy()
                    indices = (scores >= threshold)
                    if len(indices) > 0: 
                        gallery_bboxes[valid_names[i]] = boxes[indices].tolist()
                    else: 
                        gallery_bboxes[valid_names[i]] = []
        return query_bboxes, gallery_bboxes   

    def _load_or_extract_gallery_features(self):
        """ load or extract gallery features """
        if os.path.exists(self.gallery_feature_file):
            print("[1] load gallery features...")
            with open(self.gallery_feature_file, 'rb') as f:
                return pickle.load(f)
        else:
            print("[1] extract gallery features...")
            features = self._extract_gallery_features()
            # save gallery features
            with open(self.gallery_feature_file, 'wb') as f:
                pickle.dump(features, f)
            return features
        
    def _load_or_extract_query_features(self):
        """ load or extract gallery features """
        if os.path.exists(self.query_feature_file):
            print("[2] load query features...")
            with open(self.query_feature_file, 'rb') as f:
                return pickle.load(f)
        else:
            print("[2] extract query features...")
            features = self._extract_query_features()
            # save query features
            with open(self.query_feature_file, 'wb') as f:
                pickle.dump(features, f)
            return features
        
    def _extract_gallery_features(self):
        features = {}
        img_names = [f for f in os.listdir(self.gallery_path) if f.endswith('.jpg')]
        gallery_resize = transforms.Resize((746, 1000))
        for i in tqdm(range(0, len(img_names), self.batch_size), desc="extract gallery features"):
            batch_names = img_names[i:i+self.batch_size]
            batch_images = []
            valid_names = []
            num_instances = []
            
            # batch images
            for img_name in batch_names:
                img = decode_image(os.path.join(self.gallery_path, img_name))
                # img = cv2.imread(os.path.join(self.gallery_path, img_name))
                if img is not None:
                    # image preprocess
                    valid_names.append(img_name)
                    img = self.preprocess(img)
                    batch_images.append(gallery_resize(img))
                    num_instances.append(len(self.gallery_bboxes[img_name]))
            
            if not batch_images:
                continue

            new_batch_images = []
            for j, img in enumerate(batch_images):
                for bbox in self.gallery_bboxes[valid_names[j]]:
                    x1, y1, x2, y2 = map(int, bbox)
                    instance = img[y1:y2, x1:x2]
                    # instance = cv2.resize(instance, (224, 224))
                    new_batch_images.append(self.transform(instance))

            # extract features
            batch_tensor = torch.stack(new_batch_images).to(self.device)
            with torch.no_grad():
                batch_features = self.extractor(batch_tensor)

            # save features
            start = 0
            for j, img_name in enumerate(valid_names):
                gallery_features = []
                for k in range(start, start + num_instances[j]):
                    gallery_features.append(batch_features[k].cpu().numpy().flatten())
                    start += num_instances[j]
                    assert start <= batch_features.shape[0]
                features[img_name] = gallery_features
        
        print(f"#gallery images: {len(features)}")
        return features

    def _extract_query_features(self):
        img_names = [f for f in os.listdir(self.query_path) if f.endswith('.jpg')]
        features = {}

        for query_img_name in tqdm(img_names, desc="extract query features"):
            img_path = os.path.join(self.query_path, query_img_name)
            img = cv2.imread(img_path)
            assert img is not None, f'fail to load query image {query_img_name}!'
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # extract instance features
            query_features = []
            for bbox in self.query_bboxes[query_img_name]:
                x, y, w, h = bbox
                instance = img[y:y+h, x:x+w]
                # instance = cv2.resize(instance, (224, 224))
                instance_tensor = self.transform(instance).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    feature = self.extractor(instance_tensor)
                query_features.append(feature.cpu().numpy().flatten())
            
            features[query_img_name] = query_features
        print(f"#query images: {len(features)}")
        return features

    def _parse_bbox(self, txt_file):
        """ retrieve bounding box from txt_file """
        bboxes = []
        try:
            with open(txt_file, 'r') as f:
                for line in f:
                    x, y, w, h = map(int, line.strip().split())
                    bboxes.append((x, y, w, h))
        except Exception as e:
            print(f"fail to retrieve {txt_file}: {e}")
        return bboxes
    
    def _calculate_similarity(self, query_instances: list, gallery_features: dict, metric='cosine'):
        """ calculate similarity between query and gallery features """
        assert metric in ['euclidean', 'cosine'], f"Metric {metric} is not implemented!"
        similarities = {}
        for gallery_name, gallery_instances in gallery_features.items():
            if metric == 'cosine': 
                sims = [1 - cosine(q_feat, g_feat) for q_feat in query_instances for g_feat in gallery_instances]
                max_sim = np.mean(sorted(sims, reverse=True)[:len(query_instances)])
            elif metric == 'euclidean': 
                dists = [np.linalg.norm(q_feat - g_feat) for q_feat in query_instances for g_feat in gallery_instances]
                min_distances = sorted(dists)[:len(query_instances)]
                max_sim = 1 / (1 + np.mean(min_distances))  # convert distance to similarity
            else: 
                raise ValueError(f"Metric {metric} is not implemented!")
            gallery_index = int(gallery_name.split('.')[0])
            similarities[gallery_index] = max_sim
        return similarities
    
    def query(self, query_img_name, query_instances, top_k: int = 10, metric: str = 'cosine'):
        assert top_k < len(self.gallery_features)
        assert metric in ['euclidean', 'cosine'], f"Metric {metric} is not implemented!"
        # query_instances = self.query_features[query_img_name]

        similarities = self._calculate_similarity(
            query_instances  = query_instances, 
            gallery_features = self.gallery_features, 
            metric = metric)
                
        # get topK results
        query_index = int(query_img_name.split('.')[0])
        sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        top_k_sorted_results = sorted_results[:top_k]

        # log rank list
        rank_list = f'Q{query_index+1}: '
        rank_list += ' '.join([str(j) for j in [idx for idx, sim in sorted_results]])
        logging.info(rank_list)

        return top_k_sorted_results
    
    def instance_search(self, metric: str = 'cosine', top_k: int = 10, top_n: int = 5): 
        """ 
        Execute instance search for all query images.
        Visualize top K matches for top N queries. 
        """
        assert top_k < len(self.gallery_features)
        assert top_n < len(self.query_features)
        sample_gallery = {}
        for query_img_name, query_instances in tqdm(self.query_features.items(), desc='instance search'):
            top_k_results = self.query(query_img_name, query_instances, top_k=top_k, metric=metric)
            sample_gallery.setdefault(query_img_name, top_k_results)

        # visualization
        # self.visualize(sample_gallery, figname, top_k, top_n)
        return sample_gallery
    
    def visualize(self, sample_gallery, figname, top_k: int = 10, top_n: int = 5):
        """ visualize top N query results """
        # fig, axes = plt.subplots(top_n, (top_k + 1), figsize=(20, 12))
        plt.figure(figsize=(top_k + 1, top_n + 1), dpi=250)
        assert top_k < len(self.gallery_features) and top_n < len(self.query_features)
        for row in range(top_n): 
            query_img_name = f'{row}.jpg'
            query_img_path = os.path.join(self.query_path, query_img_name)
            query_img = cv2.imread(query_img_path)

            for bbox in self.query_bboxes[query_img_name]:
                x, y, w, h = bbox
                cv2.rectangle(query_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            plt.subplot(top_n, top_k+1, (top_k+1)*row+1)
            plt.imshow(cv2.resize(query_img, (224, 224)))
            plt.axis('off')
            for col in range(1, top_k+1): 
                gallery_index = sample_gallery[query_img_name][col-1][0]
                gallery_similarity = sample_gallery[query_img_name][col-1][1]
                gallery_img_path = os.path.join(self.gallery_path, f'{gallery_index}.jpg')
                gallery_img = cv2.imread(gallery_img_path)
                for bbox in self.gallery_bboxes[query_img_name]:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(query_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                plt.subplot(top_n, top_k+1, (top_k+1)*row+col+1)
                plt.title(f'{gallery_similarity:.4f}')
                plt.imshow(cv2.resize(gallery_img, (224, 224)))
                plt.axis('off')

        plt.savefig(figname) 
        plt.close()