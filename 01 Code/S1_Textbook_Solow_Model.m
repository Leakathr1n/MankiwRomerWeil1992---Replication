%% Econometric Methods I Problem Set 3
% 1_Textbook Solow Model
% Barcelona School of Economics, 2025-2026
% Author: Lea Röller

%% housekeeping
clear all; close all; clc; 

cd() %specify folder path if necessary

%% load data set
load(fullfile('..','00 Clean data','MRW1992data.mat'));

%store data in one data set to make it easier to work with
MRWdata = table(country, invest, I, N, O, pop_growth, school, Y60, Y85, Y_growth);

clear country invest I N O pop_growth school Y60 Y85 Y_growth %some housekeeping
%% clean data
% We restrict our sample for now to non-oil producing countries
MRWdata_filtered = MRWdata(MRWdata.N == 1, :);
height(MRWdata_filtered) == 98 %verify we have 98 countries left
n = height(MRWdata_filtered)

clear ans MRWdata  %some housekeeping

%% setting parameters for the analysis

format bank   % shows 2 decimals for all numeric outputs
alfa     = 0.05; %signifance level to 5%

%% Exercise 1 - Calculations
%  Replication of upper panel of MRW's Table I on p. 414.

% Equation:  log(yi) = β1 + β2log(si) + β3log(ni + g+δ)+ εi

% Generate the variables
% y = Y85 %already like this in the data set, no need to modify
MRWdata_filtered.s = MRWdata_filtered.invest / 100;           % Add s column
MRWdata_filtered.n = MRWdata_filtered.pop_growth / 100;       % Add n column
MRWdata_filtered.g_and_delta = 0.05 * ones(height(MRWdata_filtered), 1); % Add g_and_delta column
MRWdata_filtered.effective_depr = MRWdata_filtered.g_and_delta + MRWdata_filtered.n % add the effective depreciation / break-even investment rate

%taking the ln/ logs of my variables
MRWdata_filtered.log_y = log(MRWdata_filtered.Y85);               % Add log(y) column
MRWdata_filtered.log_s = log(MRWdata_filtered.s);                 % Add log(s) column
MRWdata_filtered.log_effective_depr  = log(MRWdata_filtered.effective_depr); % Add log(n + g + δ) column

%setting up my matrixes!
% X
X = [ones(height(MRWdata_filtered),1), MRWdata_filtered.log_s, MRWdata_filtered.log_effective_depr];
% y
y = MRWdata_filtered.log_y;

% Calculate beta hat
beta_hat = (X' * X) \ (X' * y);  % \ is more numerically stable than inv()

% Calculate the covariance matrix to extract the standard errors
residuals = y - X * beta_hat;          % compute residuals
k = size(X,2);                         % number of coefficients, taking out the constant!
sigma2_hat = (residuals' * residuals) / (n - k);  % estimated error variance
cov_beta = sigma2_hat * inv(X' * X);  % covariance matrix of beta_hat
se_beta = sqrt(diag(cov_beta));       % standard errors

% Calculate the R squared
y_hat = X * beta_hat; %calculate predicted values
y_bar = mean(y); % calculate mean of observed y
ESS = sum((y_hat - y_bar).^2); % Numerator: explained sum of squares
TSS = sum((y - y_bar).^2); % Denominator: total sum of squares
R2 = ESS / TSS % R-squared

% Calculate the s.e.e
s_e_e = sqrt( (residuals' * residuals) / (n - k) )

%% Exercise 1 - Export to LateX
% Export results to a file to be inserted to latex
% Define variable names and labels
varNames = {'Constant', 'ln(I/GDP)', 'ln(n+g+delta)'};
coeffs = beta_hat;
SEs = se_beta;

% File name to save
outputFolder = fullfile('..','02 Outputs');
filename = fullfile(outputFolder, 'Exercise_1.1.tex');  % full path
fid = fopen(filename, 'w'); % Open file for writing

fprintf(fid, '\\begin{table}[ht]\n');
fprintf(fid, '\\centering\n');
fprintf(fid, '\\caption{Regression Results for Non-oil Sample}\n');
fprintf(fid, '\\begin{tabular}{lcc}\n');
fprintf(fid, '\\hline\n');
fprintf(fid, 'Variable & Coefficient & Standard Error \\\\\n');
fprintf(fid, '\\hline\n');

% Loop over variables
for i = 1:length(varNames)
    fprintf(fid, '%s & %.2f & (%.2f) \\\\\n', varNames{i}, coeffs(i), SEs(i));
end

fprintf(fid, '\\hline\n');
fprintf(fid, 'Observations & \\multicolumn{2}{c}{%d} \\\\\n', n);
fprintf(fid, 'R-squared & \\multicolumn{2}{c}{%.2f} \\\\\n', R2);
fprintf(fid, 's.e.e. & \\multicolumn{2}{c}{%.2f} \\\\\n', s_e_e);
fprintf(fid, '\\hline\n');
fprintf(fid, '\\end{tabular}\n');
fprintf(fid, '\\end{table}\n');

fclose(fid);

disp(['LaTeX table exported to ', filename]);

%% Exercise 1 - Cross-Check
mdl = fitlm(MRWdata_filtered, ...
    'log_y ~ log_s + log_effective_depr');

% Display the full regression table
disp(mdl)

%%







