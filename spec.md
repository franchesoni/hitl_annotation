# app

## obj, cls tool (quick ann / active learning + show progress + save annotations)

### backend api
- put config (config)
- get config ()
- get next (strategy) <- the backend finds the id and calls get sample func
- get sample (id) 
- post annotate (id, annotation)
- delete annotate (id)
- get stats ()

### sqlite:
table config: architecture, classes
table samples: id, filepath
table predictions: id, sample_id, sample_filepath, type, class, probability, x, y, width, height, timestamp
table annotations: id, sample_id, sample_filepath, type, class, x, y, width, height, timestamp

### frontend:
navigation: next unlabeled vs. navigate seq
config button: load config and edit config
image display
classes manager: add / remove / select class
stats: plot training / performance progress

### ml:
- fastapi for training classifier
- train one epoch, predict on (2 x number of images annotated / number of epochs) images

# to do
- implement delete annotation
- implement get next func on db