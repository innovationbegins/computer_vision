
%Loading images.

fisheyedir = fullfile(toolboxdir('vision'), 'visiondata', 'fishEye');
fisheye = imageDatastore(fisheyedir);

%Display images to be stitched.

montage(fisheye.Files)

%Reading the first image from the image set.

I = readimage(fisheye, 1);

%Initializing features for I(1).

grayImage = rgb2gray(I);
points = detectSURFFeatures(grayImage);
[features, points] = extractFeatures(grayImage, points);

%Initializing all the transforms to the identity matrix.

numImages = numel(fisheye.Files);
tforms(numImages) = projective2d(eye(3));

%Initializing variable to hold image sizes.

imageSize = zeros(numImages,2);

%Iterating over remaining image pairs.

for n = 2:numImages

%Storing points and features for I(n-1).

    pointsPrevious = points;
    featuresPrevious = features;

%Reading I(n).

    I = readimage(fisheye, n);

%Converting image to grayscale.

    grayImage = rgb2gray(I);

%Saving image size.

    imageSize(n,:) = size(grayImage);

%Detecting and extract SURF features for I(n).

    points = detectSURFFeatures(grayImage);
    [features, points] = extractFeatures(grayImage, points);

%Finding correspondences between I(n) and I(n-1).

    indexPairs = matchFeatures(features, featuresPrevious, 'Unique', true);

    matchedPoints = points(indexPairs(:,1), :);
    matchedPointsPrev = pointsPrevious(indexPairs(:,2), :);

%Estimating the transformation between I(n) and I(n-1).

    tforms(n) = estimateGeometricTransform(matchedPoints, matchedPointsPrev,...
        'projective', 'Confidence', 99.9, 'MaxNumTrials', 2000);

%Computing (T(n) * T(n-1) * ...... * T(1)).

    tforms(n).T = tforms(n).T * tforms(n-1).T;
end

%Computing the output limits  for each transform.

for i = 1:numel(tforms)
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
end

avgXLim = mean(xlim, 2);

[~, idx] = sort(avgXLim);

centerIdx = floor((numel(tforms)+1)/2);

centerImageIdx = idx(centerIdx);

Tinv = invert(tforms(centerImageIdx));

for i = 1:numel(tforms)
    tforms(i).T = tforms(i).T * Tinv.T;
end

for i = 1:numel(tforms)
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
end

maxImageSize = max(imageSize);

%Finding the minimum and maximum output limits.

xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);

yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);

%Width and height of panorama.

width  = round(xMax - xMin);
height = round(yMax - yMin);

%Initializing the "empty" panorama.

panorama = zeros([height width 3], 'like', I);

blender = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaskSource', 'Input port');

%Creating a 2-D spatial reference object defining the size of the panorama.

xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width], xLimits, yLimits);

%Creating the panorama.

for i = 1:numImages

    I = readimage(fisheye, i);

%Transforming I into the panorama.

    warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);

%Generating a binary mask.

    mask = imwarp(true(size(I,1),size(I,2)), tforms(i), 'OutputView', panoramaView);

%Overlaying the warpedImage onto the panorama.

    panorama = step(blender, panorama, warpedImage, mask);
end