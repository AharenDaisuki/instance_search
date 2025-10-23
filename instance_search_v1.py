import os
import cv2
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.models as models
import torchvision.transforms as transforms

from scipy.spatial.distance import cosine
from tqdm import tqdm

class InstanceSearch:
    def __init__(self, 
                 gallery_path: str, 
                 query_path: str, 
                 txt_path: str, 
                 backbone: str = 'resnet', 
                 log_file: str = 'rankList.txt',
                 gallery_feature_file: str = "gallery_features.pkl", 
                 query_feature_file: str = "query_features.pkl",                  
                 batch_size: int = 32, ):
        assert backbone in ['resnet50','resnet101','resnet152','vgg16', 'vgg19'], f"Backbone {backbone} is not implemented!"
        if not os.path.exists(backbone):
            os.makedirs(backbone)

        self.gallery_path = gallery_path
        self.query_path = query_path
        self.txt_path = txt_path
        self.batch_size = batch_size
        self.gallery_feature_file = os.path.join(backbone, gallery_feature_file)
        self.query_feature_file = os.path.join(backbone, query_feature_file)

        # set device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.model = self._load_model(backbone=backbone)
        self.bboxes = self._load_annotation()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # log
        logging.basicConfig(filename=os.path.join(backbone, log_file), 
                            level=logging.INFO, 
                            filemode='w')
        
        # precomputed and save gallery/query features
        self.gallery_features = self._load_or_extract_gallery_features()
        self.query_features = self._load_or_extract_query_features()

    def _load_model(self, backbone):
        """ load pretrained model """
        # model factory
        if backbone == 'vgg16':
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        elif backbone == 'vgg19':
            model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        elif backbone == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        elif backbone == 'resnet101':
            model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        elif backbone == 'resnet152':
            model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
        else: 
            raise ValueError(f"Backbone {backbone} is not implemented!")
        
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model = model.to(self.device)
        model.eval()
        return model
    
    def _load_annotation(self):
        """ retrieve bounding boxes for all query images """
        bboxes_dict = {}
        img_names = [f for f in os.listdir(self.query_path) if f.endswith('.jpg')]
        for query_img_name in img_names:
            txt_file = os.path.join(self.txt_path, query_img_name.replace('.jpg', '.txt'))
            bboxes = self._parse_bbox(txt_file)
            assert len(bboxes) > 0, f'fail to load bounding box {txt_file}!'
            bboxes_dict[query_img_name] = bboxes
        return bboxes_dict      

    def _load_or_extract_gallery_features(self):
        """ load or extract gallery features """
        if os.path.exists(self.gallery_feature_file):
            print("[1] load gallery features...")
            with open(self.gallery_feature_file, 'rb') as f:
                return pickle.load(f)
        else:
            print("[1] extract gallery features...")
            features = self._extract_gallery_features_batch()
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
        
    def _extract_query_features(self):
        img_names = [f for f in os.listdir(self.query_path) if f.endswith('.jpg')]
        features = {}

        for query_img_name in tqdm(img_names, desc="extract query features"):
            img_path = os.path.join(self.query_path, query_img_name)
            img = cv2.imread(img_path)
            assert img is not None, f'fail to load query image {query_img_name}!'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # extract instance features
            query_features = []
            for bbox in self.bboxes[query_img_name]:
                x, y, w, h = bbox
                instance = img[y:y+h, x:x+w]
                instance = cv2.resize(instance, (224, 224))
                instance_tensor = self.transform(instance).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    feature = self.model(instance_tensor)
                query_features.append(feature.cpu().numpy().flatten())
            
            features[query_img_name] = query_features
        print(f"#query images: {len(features)}")
        return features

    def _extract_gallery_features_batch(self):
        """ extract gallery features in batch """
        img_names = [f for f in os.listdir(self.gallery_path) if f.endswith('.jpg')]
        features = {}
        
        for i in tqdm(range(0, len(img_names), self.batch_size), desc="extract gallery features"):
            batch_names = img_names[i:i+self.batch_size]
            batch_images = []
            valid_names = []
            
            # batch images
            for img_name in batch_names:
                img_path = os.path.join(self.gallery_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    # image preprocess
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))  # 224x224
                    batch_images.append(self.transform(img))
                    valid_names.append(img_name)
            
            if not batch_images:
                continue
                
            # batch tensors
            batch_tensor = torch.stack(batch_images).to(self.device)
            with torch.no_grad():
                batch_features = self.model(batch_tensor)
            
            # save features
            for j, img_name in enumerate(valid_names):
                features[img_name] = batch_features[j].cpu().numpy().flatten()
        
        print(f"#gallery images: {len(features)}")
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
        for gallery_name, gallery_feat in gallery_features.items():
            if metric == 'euclidean':
                dists = [np.linalg.norm(q_feat - gallery_feat) for q_feat in query_instances]
                max_sim = 1 / (1 + min(dists))  # convert distance to similarity
            elif metric == 'cosine':
                sims = [1 - cosine(q_feat, gallery_feat) for q_feat in query_instances]
                max_sim = max(sims)
                # max_sim = np.mean(sims)
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

            for bbox in self.bboxes[query_img_name]:
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
                plt.subplot(top_n, top_k+1, (top_k+1)*row+col+1)
                plt.title(f'{gallery_similarity:.4f}')
                plt.imshow(cv2.resize(gallery_img, (224, 224)))
                plt.axis('off')

        plt.savefig(figname) 
        plt.close()