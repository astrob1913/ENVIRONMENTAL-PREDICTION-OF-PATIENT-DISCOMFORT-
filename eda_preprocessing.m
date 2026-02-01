clc; 
clear all;
close all;

%% 1. LOAD DATA 
data = readtable("D:\ML_Coursework\MATLAB_scripts\Medical_Env_Discomfort_Dataset.csv");

X = data(:,1:end-1);   % 11 environmental features
Y = categorical(data.over_discomfort);  % Target (0/1)
Xmat = table2array(X);
fprintf("Rows: %d | Features: %d\n\n", size(Xmat,1), size(Xmat,2));

%% 2. SUMMARY STATISTICS 
fprintf("   Summary Statistics   \n");
summaryTable = table( ...
    X.Properties.VariableNames', ...
    mean(Xmat)', ...
    std(Xmat)', ...
    'VariableNames', {'Feature','Mean','StdDev'});
disp(summaryTable);

%% 3. CLASS IMBALANCE
fprintf("\n    Class Distribution    \n");
classCounts = countcats(Y);
disp(table(categories(Y), classCounts,'VariableNames',{'Class','Count'}));

figure;
bar(classCounts);
set(gca,'XTickLabel',categories(Y));
title('Class Distribution (Comfort vs Discomfort)');
xlabel('Class'); 
ylabel('Count');

%% 4. FEATURE HISTOGRAMS 
figure;
tiledlayout(4,3);

for i =1:width(X)
    nexttile;
    histogram(X{:,i});
    title(X.Properties.VariableNames{i},'Interpreter','none');
end
sgtitle("Histograms of All Environmental Features");

%% 5. PEARSON CORRELATION HEATMAP
corrMatrix= corr(Xmat,'Rows','pairwise');

figure;
heatmap(X.Properties.VariableNames,X.Properties.VariableNames,corrMatrix, ...
    'Colormap',redbluecmap,'CellLabelFormat','%.2f');
title("Pearson Correlation Heatmap");

%% 6. FEATURE DISTRIBUTIONS BY CLASS
figure;
tiledlayout(4,3);

for i= 1:width(X)
    nexttile; 
    hold on;
    histogram(X{Y=='0',i},'Normalization','probability','FaceAlpha',0.5);
    histogram(X{Y=='1',i},'Normalization','probability','FaceAlpha',0.5);
    title(X.Properties.VariableNames{i},'Interpreter','none');
    hold off;
end
legend("Comfort = 0","Discomfort = 1");
sgtitle("Feature Distributions Separated by Class");

%% 7. Z-SCORE NORMALISATION 
fprintf("\n   Applying Z-Score Normalisation   \n");

X_norm= (Xmat - mean(Xmat)) ./ std(Xmat);
save("normalized_features.mat","X_norm","Y");
fprintf("Normalised dataset saved as 'normalized_features.mat'.\n");
fprintf("EDA & Preprocessing complete.\n");

