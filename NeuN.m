filepaths = {"path"};
results_table = table();
for i = 1:length(filepaths)
    img = imread(filepaths{i});
    figure, imshow(img), title('Original Image'); 
    if size(img, 3) == 3
        grayImg = rgb2gray(img);
    else
        grayImg = img;
    end
    figure, imshow(grayImg), title('Grayscale Image'); 
    gaussianFilter = fspecial('gaussian', [5 5], 2);
    gaussianFilteredImg = imfilter(grayImg, gaussianFilter, 'same');
    figure, imshow(gaussianFilteredImg), title('Gaussian Filtered Image'); 
    threshold_SOX9 = entropyThreshold(gaussianFilteredImg);
    binaryImg = imbinarize(gaussianFilteredImg, threshold_SOX9);
    figure, imshow(binaryImg), title('Binarized Image');
    labeledImage = bwlabel(binaryImg);
    stats = regionprops(labeledImage, 'Area', 'PixelIdxList'); 
    finalBinaryImg = false(size(binaryImg));
    validAreas = [stats.Area];
    validIdx = validAreas >= 7 & validAreas <= 40000;

    for k = find(validIdx)
        finalBinaryImg(stats(k).PixelIdxList) = true; 
    end
    figure, imshow(finalBinaryImg), title('Final Binary Image with Valid Areas'); 

    totalValidArea = sum(validAreas(validIdx));  
    imageCoveragePercent = (totalValidArea / numel(binaryImg)) * 100;  
    newRow = table({filepaths{i}}, totalValidArea, imageCoveragePercent, ...
                   'VariableNames', {'Filepath', 'Total_Valid_Area', 'Coverage_Percent'});
    results_table = [results_table; newRow];
    disp(['Total Valid Area (pixels): ', num2str(totalValidArea)]);
    disp(['Coverage Percent: ', num2str(imageCoveragePercent), '%']);
end
save('SOX9_analysis_results.mat', 'results_table');

function threshold = entropyThreshold(I)
    I = im2double(I);
    [counts, ~] = imhist(I);
    p = counts / sum(counts);
    maxEntropy = -Inf;
    threshold = 0;

    for t = 1:length(p) - 1
        background = p(1:t);
        object = p(t + 1:end);
        background = background / sum(background);
        object = object / sum(object);
        backgroundEntropy = -nansum(background .* log2(background + eps));
        objectEntropy = -nansum(object .* log2(object + eps));
        totalEntropy = backgroundEntropy + objectEntropy;

        if totalEntropy > maxEntropy
            maxEntropy = totalEntropy;
            threshold = t;
        end
    end

    threshold = (double(threshold) / 255)*0.81;  
end

