# POS-tagging

* Flair Embeddings

Predict:
`predict.py --pos_checkpoint_path <path to checkpoint> --vocab <path to vocabulary file> --data <sentence to predict POS tags for>`

Train:
`train.py --data <path to the data archive> --checkpoint_path <path to the directory to save checkpoints to> --vocab <path to a vocab directory> --epochs <number of epochs> --batch_size <batch size>`