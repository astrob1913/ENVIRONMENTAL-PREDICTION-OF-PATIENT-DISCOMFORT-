clc; 
clear; 
close all;

%% 1. LOAD MODELS & TEST DATA 
load("LR_best.mat");   % Trained Logistic Regression
load("RF_best.mat");   % Trained Random Forest
load("test_data.mat"); 
load("CV_metrics.mat");  
load("LR_test_metrics.mat");
load("RF_test_metrics.mat");

%% 2. LOGISTIC REGRESSION ROC 
[~, LR_scores]= predict(LR_best, Xtest);
LR_probs= LR_scores(:,2);   % Probability of class '1'

[LR_X, LR_Y, ~, LR_AUC]= perfcurve(Ytest, LR_probs,'1');

%% 3. RANDOM FOREST ROC 
[~, RF_scores] = predict(RF_best, Xtest);
RF_probs = RF_scores(:,2);                      % Probability of class '1'
[RF_X, RF_Y, ~, RF_AUC]= perfcurve(Ytest, RF_probs, '1');

%% 4. COMBINED ROC PLOT 
figure;
plot(LR_X, LR_Y,'LineWidth', 2); hold on;
plot(RF_X,RF_Y, 'LineWidth', 2);
plot([0 1],[0 1],'k--');          % Random classifier baseline

legend( ...
    sprintf('Logistic Regression (AUC = %.3f)', LR_AUC), ...
    sprintf('Random Forest (AUC = %.3f)', RF_AUC), ...
    'Location','southeast');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve Comparison: Logistic Regression vs Random Forest');
grid on;

%%  5. DISPLAY AUC VALUES 
fprintf("\n    TEST SET AUC COMPARISON    \n");
fprintf("Logistic Regression AUC = %.3f\n", LR_AUC);
fprintf("Random Forest AUC       = %.3f\n", RF_AUC);

%% OVERFITTING CHECK 

figure;
metrics = {'Precision','Recall','F1-score'};
values = [ ...
    LR_CV_Prec  LR_Test_Prec;
    LR_CV_Rec   LR_Test_Rec;
    LR_CV_F1    LR_Test_F1 ];

b = bar(values, 'grouped');
b(1).BarWidth = 0.9;
b(2).BarWidth = 0.9;
set(gca,'XTickLabel',metrics)
legend({'10-CV','Test'},'Location','northeast')
title('LR Overfitting Check (CV vs Test)')
ylabel('Score')
grid on
% Zoom Y-axis to highlight differences
ylim([min(values(:))-0.03 1])

% Add value labels
for i = 1:size(values,1)
    for j = 1:2
        text(i + (j-1.5)*0.15, values(i,j), ...
            sprintf('%.3f',values(i,j)), ...
            'HorizontalAlignment','center', ...
            'VerticalAlignment','bottom', ...
            'FontSize',9);
    end
end

figure;
metrics= {'Precision','Recall','F1-score'};
values= [ ...
    RF_CV_Prec  RF_Test_Prec;
    RF_CV_Rec   RF_Test_Rec;
    RF_CV_F1    RF_Test_F1 ];

b = bar(values, 'grouped');
b(1).BarWidth = 0.9;
b(2).BarWidth = 0.9;
set(gca,'XTickLabel',metrics)
legend({'10-CV','Test'}, 'Location','northwest')
title('RF Overfitting Check (CV vs Test)')
ylabel('Score')
grid on

ylim([min(values(:))-0.03 1])
for i = 1:size(values,1)
    for j = 1:2
        text(i + (j-1.5)*0.15, values(i,j), ...
            sprintf('%.3f',values(i,j)), ...
            'HorizontalAlignment','center', ...
            'VerticalAlignment','bottom', ...
            'FontSize',9);
    end
end
