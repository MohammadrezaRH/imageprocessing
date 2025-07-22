
filepaths = {"path"};
results_table = table();
for i = 1:length(filepaths)
    img = imread(filepaths{i});
    if size(img, 3) == 3
        grayImg = rgb2gray(img);
    else
        grayImg = img;
    end
    se = strel('disk', 1000); 
    background = imopen(grayImg, se);
    subtractedImg = imsubtract(grayImg, background);
    gaussianFilter = fspecial('gaussian', [5 5], 1); 
    gaussianFilteredImg = imfilter(subtractedImg, gaussianFilter, 'same');
    threshold_SOX9 = entropyThreshold(gaussianFilteredImg);
    binaryImg = imbinarize(gaussianFilteredImg, threshold_SOX9);
    binaryImg = bwareaopen(binaryImg, 5);
    labeledImage = bwlabel(binaryImg);
    figure, imshow(binaryImg), title('Labeled Image');
    stats = regionprops(labeledImage, 'Area', 'Perimeter');
    imageArea = numel(grayImg);  
    sox9_positive_count = 0;
    for k = 1:length(stats)
        if stats(k).Area <= 10000
            sox9_positive_count = sox9_positive_count + 1;
        end
    end
    cell_density = sox9_positive_count / imageArea;
    newRow = table({filepaths{i}}, sox9_positive_count, imageArea, cell_density, 'VariableNames', {'Filepath', 'Cell_Count', 'Image_Area', 'Density'});
    results_table = [results_table; newRow];
    disp(['Number of SOX9 Positive Cells: ', num2str(sox9_positive_count)]);
    disp(['Total Image Area (pixels): ', num2str(imageArea)]);
    disp(['Density of SOX9 Positive Cells: ', num2str(cell_density), ' cells per pixel']);
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

    threshold = (double(threshold) / 255); 
end

