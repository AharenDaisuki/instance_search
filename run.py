import argparse
from instance_search_base import InstanceSearch
from instance_search_vgg import InstanceSearchVGG
from instance_search_vit import InstanceSearchViT

def instance_search_factory(method):
    if method in ['resnet50', 'resnet101', 'resnet152', 
                  'efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l']:
        searcher = InstanceSearch
    elif method in ['vgg16', 'vgg19']: 
        searcher = InstanceSearchVGG
    elif method in ['vit_b_16']:
        searcher = InstanceSearchViT
    else: 
        raise ValueError(f"Method {method} is not implemented!")
    return searcher

def main(args):
    # configuration
    gallery_path = args.gallery
    query_path = args.query
    txt_path = args.annotation
    backbone = args.method
    batch_size = args.batch_n
    top_k = args.top_k
    top_n = args.top_n  
    figname = args.output
    
    # initialization
    model = instance_search_factory(backbone)
    searcher = model(gallery_path, query_path, txt_path, backbone=backbone, batch_size=batch_size)

    # instance search 
    sample_gallery = searcher.instance_search(top_k=top_k, top_n=top_n)

    # visualization
    searcher.visualize(sample_gallery = sample_gallery, 
                       figname = figname, 
                       top_k = top_k, 
                       top_n = top_n,)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method',  type = str, default = 'resnet50', help = 'Instance search method')
    parser.add_argument('--output',  type = str, default = 'demo.png', help = 'Output image name')
    parser.add_argument('--gallery', type = str, default = 'gallery/gallery', help = 'Gallery directory')
    parser.add_argument('--query',   type = str, default = 'gallery/query', help = 'Query directory')
    parser.add_argument('--annotation', type = str, default = 'query_txt', help = 'Annotation directory')
    parser.add_argument('--top_k',   type = int, default = 10, help = 'Top K matched images')
    parser.add_argument('--top_n',   type = int, default = 5, help = 'Top N query visualizations')
    parser.add_argument('--batch_n', type = int, default = 64, help = 'Batch size for feature extraction') 
    args = parser.parse_args()
    main(args)