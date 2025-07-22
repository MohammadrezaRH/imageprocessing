

close all;
filepaths = {"path"}
results_table = table();
for i = 1:length(filepaths)
    img = imread(filepaths{i});
    if size(img, 3) == 3
        grayImg = rgb2gray(img);
    else
        grayImg = img;
    end
    figure, imshow(grayImg), title('Grayscale Image');
    se = strel('disk', 170); 
    background = imopen(grayImg, se);
    subtractedImg = imsubtract(grayImg, background);
    figure, imshow(subtractedImg), title('Grayscale Image')
    gaussianFilter = fspecial('gaussian', [5 5], 1); 
    gaussianFilteredImg = imfilter(subtractedImg, gaussianFilter, 'same');
    threshold_SOX9 = entropyThreshold(gaussianFilteredImg);
    binaryImg = imbinarize(gaussianFilteredImg, threshold_SOX9);
    binaryImg = bwareaopen(binaryImg, 0);
    figure, imshow(binaryImg), title('Cleaned Binary Image');

 labeledImage = bwlabel(binaryImg);
    figure, imshow(label2rgb(labeledImage)), title('Labeled Image');
    stats = regionprops(labeledImage, 'Area', 'Perimeter');
    imageArea = numel(grayImg); 
    sox9_positive_count = 0;
    sox9_positive_area = 0;
    for k = 1:length(stats)
        if stats(k).Area <= 10000
            sox9_positive_count = sox9_positive_count + 1;
            sox9_positive_area = sox9_positive_area + stats(k).Area;
        end
    end
    normalized_area_coverage = sox9_positive_area / imageArea;
    newRow = table({filepaths{i}}, sox9_positive_count, sox9_positive_area, imageArea, normalized_area_coverage, 'VariableNames', {'Filepath', 'Cell_Count', 'SOX9_Positive_Area', 'Image_Area', 'Normalized_Area_Coverage'});
    results_table = [results_table; newRow];
    disp(['GFAP Positive Area (pixels): ', num2str(sox9_positive_area)]);
    disp(['Total Image Area (pixels): ', num2str(imageArea)]);
    disp(['Normalized Area Coverage: ', num2str(normalized_area_coverage)]);
end
save('SOX9_analysis_results.mat', 'results_table');
disp(results_table);
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

    threshold = (double(threshold) / 255); 
end


