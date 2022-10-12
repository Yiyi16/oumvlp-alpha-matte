# Generate Alpha Matte for OUMVLP

### Preprocessing step to obtain trimap
Trimap is the user defined area where the alpha matte is estimated.

Given the coarse segmentation mask, we obtain trimap by eroding both the fg area (subject area) and bg area.
[!image](examples/RGB/00005/030_00/0033.png)
[!image](examples/sil/00005/030_00/0033.png)
[!image](examples/trimap/00005/030_00/0033.png)
<p align="center">
  <img src="examples/RGB/00005/030_00/0033.png" width="160" title="Original Image"/>
  <img src="examples/sil/00005/030_00/0033.png" width="160" title="segmentation Mask"/>
  <img src="examples/trimap/00005/030_00/0033.png" width="160" title="Trimap"/>
</p>

'fg_kernel' is the required parameter in the matrix describing how the foreground area of the coarse segmentation mask is eroded as the defined foreground. 

'bg_kernel' works in a similar way.


```bash
python gen_trimap.py 
```

