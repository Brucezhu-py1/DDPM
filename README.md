# DDPM
For generating microstructure images under specific conditions.

Training Data: 1157 grayscale images at 1036×1103 pixels.

Process:
1. The experimental material in this study is forged T55511 titanium alloy (nominal composition Ti-5Al-5Mo-5V-1Cr-1Fe).
2. Samples were taken from the forged bar, with the size of the samples being 8×8×10 mm³.
3. The microstructure observation samples and the corresponding tensile samples were heat treated together, using the STA heat treatment process.
4. Samples forged at 250°C were also subjected to two different conditions, held at 1°C for 5.2 hours or 350 hours, and then cooled to the forging temperature during the remaining soaking time.
5. The solution treatment process used 10°C intervals, with 16 solution temperatures from 710°C to 860°C, holding time of 2 hours; the aging process was at two temperatures, 500°C and 550°C, holding time of 3 hours, with air cooling as the cooling method for the experiments.
6. The factor of β phase grain size was not considered when constructing the model data in this study.
7. The reason is that the solution temperature used in the STA process of this study is below the α/β temperature, where the αp grains will pin the β grains, so the β phase grain size is relatively stable throughout the entire STA process used in this study, hence this factor is not considered as a variable in this model study.
8. For the microstructure samples after STA heat treatment, first, samples for microstructure observation were ground on silicon carbide (SiC) sandpapers with grits of 80, 400, 1000, and 2000, which were wetted with water in sequence.
9. Then, electrochemical polishing was carried out in a polishing solution of methanol: butanol: perchloric acid = 6:3:1 at a liquid environment of -30°C. The SEM samples were etched in a solution of HF:HNO3:H2O=3:7:90.
10. The microstructure characteristics under different heat treatment processes were studied using the FEI Quanta 650 FEG scanning electron microscope (SEM). The image magnification ranged from 5,000 to 100,000 to ensure data diversity, ultimately forming 1157 SEM microstructure image data at different magnifications of 1536×1103.

Data Processing:
1. Prepare the training data, at the beginning of each iteration, the actual 1036×1103 images were randomly cropped to 512×512 image size.
2. The image magnification ranges from 5,000 to 100,000 to ensure data diversity, and the data were finally divided into 23 different categories.
3. During the training period, 4 image categories were set aside, accounting for 17% of the data. This resulted in a 17:83 test/train split, which is different from the more common 20:80 or 10:90 data partitioning.
4. Since microstructure images are indifferent to rotation, random horizontal and vertical flips were applied to the cropped images with a probability of 0.5 as a data augmentation technique.

Model Implementation:
1. The DDPM generative model was implemented in Python with the help of the PyTorch library.
2. The U-net architecture for image segmentation was used to find the noise distribution introduced.
3. The encoding part consists of six max pooling blocks, each followed by four convolutional layers, group normalization, and Gaussian Error Linear Unit (GELU) activation functions. Self-attention mechanisms with four heads were applied after each down-sampling layer to help achieve long-distance and multi-level dependencies across image regions.
4. The decoding part is the mirror structure of the encoding. It consists of six modules with bilinear up-sampling, each followed by four convolutional layers, group normalization, and GELU activation functions.
5. The only difference between the decoding and encoding architecture is that, after each up-sampling layer, the data is connected to the skip connections corresponding to the channels of the same resolution in the encoding part to prevent the loss of spatial information in the encoder layers and to alleviate the vanishing gradient problem. The model is regulated by the Transformer position embedding time step and added to each down-sampling and up-sampling layer.
6. To regulate the model according to several specific process parameters, the following method was proposed. At each down-sampling layer and up-sampling layer, the embedding vector corresponding to the process parameters is linearly mapped to a separate tensor of the same size as the corresponding layer's spatial dimensions, and then connected to the feature map of that layer. The Adam optimizer and Mean Squared Error (MSE) loss function were used for model training.
7. Exponential Moving Average (EMA) technique was used to compute the exponential weighted average of the current and updated model parameters at each optimization step to stabilize the optimization process.
8. The model was trained on a 4090 GPU.
9. After 300 iterations, equivalent to over 24 hours of computation with our available resources, the training was stopped.

Image Evaluation:
1. The quality of the synthesized images was quantitatively assessed using the MIPAR software for contrast estimation.

Input and Output:
1. The DDPM model is based on inputs of STA process parameters, performance, and other process parameters.
2. High-resolution SEM microstructure images of the forged AZ80 magnesium alloy were generated.
