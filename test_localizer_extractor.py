import argparse
from instance_search import InstanceSearch

def main(args):
    # configuration
    gallery_path = args.gallery
    query_path = args.query
    txt_path = args.annotation
    localizer = args.localizer
    extractor = args.extractor
    batch_size = args.batch_n
    top_k = args.top_k
    top_n = args.top_n  
    figname = args.output
    
    # initialization
    searcher = InstanceSearch(gallery_path, query_path, txt_path, localizer=localizer, extractor=extractor, batch_size=batch_size)

    # instance search 
    sample_gallery = searcher.instance_search(top_k=top_k, top_n=top_n)

    # visualization
    searcher.visualize(sample_gallery = sample_gallery, 
                       figname = figname, 
                       top_k = top_k, 
                       top_n = top_n,)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--extractor',  type = str, default = 'resnet152', help = 'Feature extraction method')
    parser.add_argument('--localizer',  type = str, default = 'fasterrcnn', help = 'Instance localization method')
    parser.add_argument('--output',  type = str, default = 'demo.png', help = 'Output image name')
    parser.add_argument('--gallery', type = str, default = 'gallery/gallery', help = 'Gallery directory')
    parser.add_argument('--query',   type = str, default = 'gallery/query', help = 'Query directory')
    parser.add_argument('--annotation', type = str, default = 'query_txt', help = 'Annotation directory')
    parser.add_argument('--top_k',   type = int, default = 10, help = 'Top K matched images')
    parser.add_argument('--top_n',   type = int, default = 5, help = 'Top N query visualizations')
    parser.add_argument('--batch_n', type = int, default = 32, help = 'Batch size for feature extraction') 
    args = parser.parse_args()
    main(args)
    