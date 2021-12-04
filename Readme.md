# CIRCUS CS Plug-in Samples

This repository contains source codes of Extract visceral fat (VAT) / subcutaneous fat (SAT) region in whole body Dixon MR (CIRCUS CS plug-ins).

**Caution!!**

This repository does not contain a file of the segmentation model (model_random_search_best.pth).

## Build and Install a Plug-in

    $ cd mr_fat_volumetry
    $ docker build -t circus/dummy-mr_fat_volumetry:1.0.0-beta .

    $ cd /path/to/circus-api
    $ node circus register-cad-plugin <full-image-id>

The full image ID (sha256) can be obtained by:

    $ docker image ls --no-trunc