import os 
import torch
from tqdm import tqdm 
from PIL import Image
from instance_search_base import InstanceSearch

def extract_cls_token(features, layer_name):
    return features[layer_name][:, 0]

class InstanceSearchViT(InstanceSearch): 
    def __init__(self, 
                 gallery_path, 
                 query_path, 
                 txt_path, 
                 backbone = 'vit_l_16', 
                 log_file = 'rankList.txt', 
                 gallery_feature_file = "gallery_features.pkl", 
                 query_feature_file = "query_features.pkl", 
                 batch_size = 16):
        super().__init__(gallery_path, query_path, txt_path, backbone, 
                         log_file, gallery_feature_file, query_feature_file, batch_size)
        
    def _extract_query_features(self):
        img_names = [f for f in os.listdir(self.query_path) if f.endswith('.jpg')]
        features = {}

        for query_img_name in tqdm(img_names, desc="extract query features"):
            img_path = os.path.join(self.query_path, query_img_name)
            img = Image.open(img_path).convert('RGB')
            
            # extract instance features
            query_features = []
            for bbox in self.bboxes[query_img_name]:
                x, y, w, h = bbox
                instance = img.crop((x, y, x+w, y+h))
                instance_tensor = self.transform(instance).unsqueeze(0).to(self.device)
                
                # inference
                with torch.no_grad():
                    feature = self.model(instance_tensor)
                    cls_token = extract_cls_token(feature, 'final_transformer_block')
                query_features.append(cls_token.cpu().numpy().flatten())
            
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
            
            # batch images
            for img_name in batch_names:
                img_path = os.path.join(self.gallery_path, img_name)
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                batch_images.append(img)
            
            if not batch_images:
                continue
                
            # batch tensors
            batch_tensor = torch.stack(batch_images).to(self.device)
            with torch.no_grad():
                batch_features = self.model(batch_tensor)
            
            # save features
            for j, img_name in enumerate(batch_names):
                cls_tokens = extract_cls_token(batch_features, 'final_transformer_block')
                features[img_name] = cls_tokens[j].cpu().numpy().flatten()

        print(f"#gallery images: {len(features)}")
        return features