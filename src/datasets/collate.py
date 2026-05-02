def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def inference_collate_fn(batch):
    images, metas = zip(*batch)
    return list(images), list(metas)
