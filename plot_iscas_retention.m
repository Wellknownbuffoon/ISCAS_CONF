function plot_iscas_retention(dataDir, outputDir)
%PLOT_ISCAS_RETENTION Reproduce both figures and the adapted research table.
% MATLAB R2020a or newer; no additional toolboxes required.
%
% Unzip the supplied package, open that folder in MATLAB, then run:
%   plot_iscas_retention
%
% Optional explicit folders:
%   plot_iscas_retention('C:\my_data', 'C:\my_figures')
%
% Input: source_fold_means.csv (already averaged over 30 realizations/fold).
% Error = 100 - accuracy; bands = mean +/- 2 SAMPLE SD across fold means.
% Table III compares HWA and NSR-HWA with the same DriftSlice B=1 correction.
% The x-axis uses equally spaced timestamp positions, as in the screenshots.
% No smoothing, curve fitting, training, or parameter tuning is performed.

if nargin < 1 || isempty(dataDir)
    dataDir = fileparts(mfilename('fullpath'));
end
if nargin < 2 || isempty(outputDir)
    outputDir = fullfile(dataDir, 'matlab_figures');
end
inputFile = fullfile(dataDir, 'source_fold_means.csv');
assert(isfile(inputFile), 'Missing input CSV: %s', inputFile);
T = readtable(inputFile, 'TextType', 'string');
required = {'method', 'compensation', 't_index', 't_s', 'fold', 'accuracy'};
assert(all(ismember(required, T.Properties.VariableNames)), ...
    'The input CSV is missing required columns.');
T.method = string(T.method);
T.compensation = string(T.compensation);

methods = ["PCM-HWA", "PCM-NSR-HWA"];
compensations = ["none", "GDC-safe", "DS-B1", "DS-val"];
T = T(ismember(T.method, methods) & ismember(T.compensation, compensations), :);
assert(~isempty(T), 'No HWA/NSR data found.');
assert(all(isfinite(T.accuracy) & T.accuracy >= 0 & T.accuracy <= 100), ...
    'Accuracy must be finite and expressed in percent (0 to 100).');
keys = {'method', 'compensation', 't_index', 'fold'};
assert(height(unique(T(:, keys))) == height(T), 'Duplicate fold-mean records.');

day = 86400;
year = 365.25 * day;
times = [20, 60, 1800, 3600, day, 5*day, 30*day, 180*day, ...
         year, 2*year, 3*year, 5*year];
labels = {'20 s', '1 min', '30 min', '1 h', '1 d', '5 d', ...
          '1 mo', '6 mo', '1 y', '2 y', '3 y', '5 y'};
assert(isequal(unique(T.t_index), (0:11)'), 'Expected all 12 retention timestamps.');
folds = unique(T.fold);
F = numel(folds);
assert(F >= 2, 'At least two training repetitions are needed for sample SD.');

% Recompute all displayed values from fold means, never pooled raw SD.
n = numel(methods) * numel(compensations) * numel(times);
S = table('Size', [n, 8], ...
    'VariableTypes', {'string','string','double','double','double','double','double','double'}, ...
    'VariableNames', {'method','compensation','t_index','t_s', ...
                      'mean_accuracy','sd_across_folds','fold_count','mean_test_error'});
k = 0;
for m = 1:numel(methods)
    for c = 1:numel(compensations)
        for ti = 0:11
            A = T(T.method == methods(m) & T.compensation == compensations(c) ...
                  & T.t_index == ti, :);
            assert(height(A) == F && isequal(unique(A.fold), folds), ...
                'Missing or unmatched folds for %s / %s / timestamp %d.', ...
                char(methods(m)), char(compensations(c)), ti);
            assert(all(abs(A.t_s - times(ti+1)) < 1e-6), 'Retention time mismatch.');
            k = k + 1;
            mu = mean(A.accuracy);
            sd = std(A.accuracy, 0);  % N-1 denominator: sample SD.
            S(k,:) = {methods(m), compensations(c), ti, times(ti+1), mu, sd, F, 100-mu};
        end
    end
end

black = [17,17,17]/255;
orange = [213,94,0]/255;
blue = [0,114,178]/255;
green = [0,158,115]/255;

% ---------------- Figure 1: test error with +/-2 SD bands ----------------
fig1 = figure('Color','w', 'Name','PCM drift: test error', ...
              'Position',[80,80,1600,640]);
ax1 = axes('Parent',fig1, 'Position',[0.065,0.255,0.43,0.61]);
ax2 = axes('Parent',fig1, 'Position',[0.54,0.255,0.43,0.61]);
for ax = [ax1, ax2]
    hold(ax, 'on');
    set(ax, 'FontName','Arial', 'FontSize',12, 'Box','off', ...
        'XLim',[0.45,12.55], 'YLim',[9.1,17.1], 'YTick',10:17, ...
        'XTick',1:12, 'XTickLabel',labels, 'XTickLabelRotation',45, ...
        'XGrid','on', 'YGrid','on', 'GridLineStyle',':', 'GridAlpha',0.20);
    xlabel(ax, 'Time after programming', 'FontSize',14);
end
ylabel(ax1, 'Test error (%)', 'FontSize',14);
set(ax2, 'YTickLabel', []);
title(ax1, '(a) NSR training comparison: DriftSlice B=1', 'FontSize',14, 'Interpreter','none');
title(ax2, '(b) Compensation comparison: PCM-HWA', 'FontSize',14, 'Interpreter','none');

h1 = drawCurve(ax1, S, "PCM-HWA", "DS-B1", black);
h2 = drawCurve(ax1, S, "PCM-NSR-HWA", "DS-B1", orange);
uistack([h1,h2], 'top');
legend(ax1, [h1,h2], {'HWA + DriftSlice B=1','NSR-HWA + DriftSlice B=1'}, ...
    'Location','northwest', 'FontSize',11, 'Interpreter','none');

h3 = drawCurve(ax2, S, "PCM-HWA", "GDC-safe", black);
h4 = drawCurve(ax2, S, "PCM-HWA", "DS-B1", blue);
h5 = drawCurve(ax2, S, "PCM-HWA", "DS-val", green);
uistack([h3,h4,h5], 'top');
legend(ax2, [h3,h4,h5], ...
    {'HWA + GDC-safe','HWA + DriftSlice B=1','HWA + DriftSlice selected B'}, ...
    'Location','northwest', 'FontSize',11, 'Interpreter','none');
addText(fig1, [0.03,0.925,0.94,0.06], ...
    'ResNet-32 / CIFAR-10: test error under PCM drift', 19);
addText(fig1, [0.03,0.005,0.94,0.09], ...
    {sprintf('Mean +/- 2 SD across %d training repetitions; 30 device realizations per repetition', F), ...
     'Timestamps are equally spaced on the x-axis; bands describe variability, not 95% confidence intervals'}, 11);

% ---------------- Figure 2: split-triangle endpoint heatmap --------------
% Each difference is candidate accuracy minus control accuracy (pp).
H = table('Size',[8,7], ...
    'VariableTypes',{'string','string','string','string','string','double','double'}, ...
    'VariableNames',{'panel','candidate_method','candidate_compensation', ...
                     'control_method','control_compensation', ...
                     'delta_accuracy_20s_pp','delta_accuracy_5y_pp'});
leftInitial = zeros(1,4); leftFinal = zeros(1,4);
rightInitial = zeros(2,2); rightFinal = zeros(2,2);
for c = 1:4
    [a,b] = endpointDifference(S, "PCM-NSR-HWA", compensations(c), "PCM-HWA", compensations(c));
    leftInitial(c) = a; leftFinal(c) = b;
    H(c,:) = {"NSR minus HWA (same compensation)", "PCM-NSR-HWA", ...
        compensations(c), "PCM-HWA", compensations(c), a, b};
end
k = 4;
for m = 1:2
    for c = 1:2
        ds = compensations(c+2);
        [a,b] = endpointDifference(S, methods(m), ds, methods(m), "GDC-safe");
        rightInitial(m,c) = a; rightFinal(m,c) = b;
        k = k + 1;
        H(k,:) = {"DriftSlice minus GDC-safe (same training)", ...
                  methods(m), ds, methods(m), "GDC-safe", a, b};
    end
end
limit = max(abs([H.delta_accuracy_20s_pp; H.delta_accuracy_5y_pp]));
if limit == 0, limit = 1; end
map = interp1([0,0.5,1], [178,24,43;247,247,247;27,120,55]/255, linspace(0,1,257));
fig2 = figure('Color','w', 'Name','Initial versus five-year accuracy differences', ...
              'Position',[100,100,1600,550]);
ax3 = axes('Parent',fig2, 'Position',[0.08,0.32,0.50,0.34]);
ax4 = axes('Parent',fig2, 'Position',[0.65,0.17,0.25,0.68]);
colormap(fig2, map);
drawTriangles(ax3, leftInitial, leftFinal, limit, ...
    {'NSR - HWA'}, {'None','GDC-safe','DS-B1','DS-val'});
drawTriangles(ax4, rightInitial, rightFinal, limit, ...
    {'HWA','NSR-HWA'}, {'DS-B1','DS-val'});
title(ax3, 'NSR minus HWA (same compensation)', 'FontSize',14, 'Interpreter','none');
title(ax4, 'DriftSlice minus GDC-safe (same training)', 'FontSize',14, 'Interpreter','none');
cb = colorbar(ax4);
set(ax4, 'Position',[0.65,0.17,0.25,0.68]);
set(cb, 'Position',[0.922,0.17,0.012,0.68], 'FontSize',12);
cb.Label.String = 'Accuracy change (percentage points)';
cb.Label.FontSize = 13;
addText(fig2, [0.03,0.935,0.94,0.06], ...
    'Full-run accuracy differences: initial versus five-year retention', 19);
addText(fig2, [0.03,0.005,0.94,0.10], ...
    {'Upper triangle: 20 s; Lower triangle: 5 y; Green: improvement; Red: degradation', ...
     sprintf('Differences of means across %d training repetitions; colors do not indicate statistical significance.', F)}, 11);

if ~isfolder(outputDir), mkdir(outputDir); end
writetable(S, fullfile(outputDir, 'matlab_aggregated_source.csv'));
writetable(H, fullfile(outputDir, 'matlab_heatmap_source.csv'));
saveFigure(fig1, outputDir, 'figure1_test_error');
saveFigure(fig2, outputDir, 'figure2_split_triangle_heatmap');
[fig3, paperTable] = drawRetentionTable(S);
writetable(paperTable, fullfile(outputDir, 'table_III_drift_robustness.csv'));
writeLatexTable(paperTable, fullfile(outputDir, 'table_III_drift_robustness.tex'));
saveFigure(fig3, outputDir, 'table_III_drift_robustness');
disp(H);
disp(paperTable);
fprintf('\nSaved both figures and Table III as PNG, vector PDF and editable FIG in:\n%s\n', outputDir);
end


function h = drawCurve(ax, S, method, compensation, color)
A = sortrows(S(S.method == method & S.compensation == compensation,:), 't_index');
x = (1:height(A))';
y = A.mean_test_error;
sd = A.sd_across_folds;
fill(ax, [x;flipud(x)], [y-2*sd;flipud(y+2*sd)], color, ...
    'FaceAlpha',0.14, 'EdgeColor','none', 'HandleVisibility','off');
h = plot(ax, x, y, '-o', 'Color',color, 'LineWidth',2, ...
    'MarkerSize',5, 'MarkerFaceColor',color);
end


function [initial, final] = endpointDifference(S, cm, cc, bm, bc)
C = sortrows(S(S.method == cm & S.compensation == cc,:), 't_index');
B = sortrows(S(S.method == bm & S.compensation == bc,:), 't_index');
initial = C.mean_accuracy(1) - B.mean_accuracy(1);
final = C.mean_accuracy(end) - B.mean_accuracy(end);
end


function drawTriangles(ax, initial, final, limit, rowLabels, colLabels)
hold(ax, 'on');
[nr,nc] = size(initial);
for r = 1:nr
    for c = 1:nc
        x = c-1; y = r-1;
        % YDir='reverse': these vertices give the upper-right triangle.
        patch(ax, [x,x+1,x+1], [y,y,y+1], initial(r,c), ...
            'FaceColor','flat', 'CDataMapping','scaled', 'EdgeColor',[0.35,0.35,0.35], 'LineWidth',0.8);
        patch(ax, [x,x,x+1], [y,y+1,y+1], final(r,c), ...
            'FaceColor','flat', 'CDataMapping','scaled', 'EdgeColor',[0.35,0.35,0.35], 'LineWidth',0.8);
        addValue(ax, x+0.72, y+0.26, initial(r,c), limit);
        addValue(ax, x+0.28, y+0.74, final(r,c), limit);
    end
end
set(ax, 'YDir','reverse', 'XLim',[0,nc], 'YLim',[0,nr], ...
    'XTick',(0:nc-1)+0.5, 'XTickLabel',colLabels, ...
    'YTick',(0:nr-1)+0.5, 'YTickLabel',rowLabels, ...
    'TickLength',[0,0], 'FontName','Arial', 'FontSize',13, 'Box','off');
axis(ax, 'equal');
caxis(ax, [-limit,limit]);
end


function addValue(ax, x, y, value, limit)
color = [0.07,0.07,0.07];
if abs(value) > 0.7*limit, color = [1,1,1]; end
text(ax, x, y, sprintf('%+.2f',value), 'HorizontalAlignment','center', ...
    'VerticalAlignment','middle', 'FontName','Arial', 'FontSize',14, ...
    'Color',color, 'Interpreter','none');
end


function addText(fig, position, value, fontSize)
annotation(fig, 'textbox', position, 'String',value, ...
    'EdgeColor','none', 'HorizontalAlignment','center', 'VerticalAlignment','middle', ...
    'FontName','Arial', 'FontSize',fontSize, 'Interpreter','none');
end


function saveFigure(fig, outputDir, name)
drawnow;
exportgraphics(fig, fullfile(outputDir, [name,'.png']), 'Resolution',300, 'BackgroundColor','white');
exportgraphics(fig, fullfile(outputDir, [name,'.pdf']), 'ContentType','vector', 'BackgroundColor','white');
savefig(fig, fullfile(outputDir, [name,'.fig']));
end


function [fig, P] = drawRetentionTable(S)
% Adapt the requested table to the actual NSR experiment, not the reference
% paper's DAR/time-grid search. Compensation is held fixed at DriftSlice B=1.
% Final Test before PCM is unavailable in these CSVs: keep it NaN / em dash.
methodNames = ["PCM-HWA"; "PCM-NSR-HWA"];
trainingNames = ["Standard HWA"; "NSR-HWA"];
lambda = [0; 0.1];  % Configured NSR coefficient of the supplied completed run.
P = table(repmat("ResNet-32",2,1), trainingNames, repmat("DS-B1",2,1), lambda, ...
    nan(2,1), zeros(2,1), zeros(2,1), zeros(2,1), ...
    nan(2,1), nan(2,1), nan(2,1), nan(2,1), ...
    'VariableNames', {'model','training','compensation','lambda_nsr', ...
    'final_test_accuracy','initial_20s_accuracy','final_5y_accuracy','max_drop_pp', ...
    'improvement_1h_pp','improvement_1d_pp','improvement_6mo_pp','improvement_5y_pp'});
B = sortrows(S(S.method == "PCM-HWA" & S.compensation == "DS-B1",:), 't_index');
for m = 1:2
    A = sortrows(S(S.method == methodNames(m) & S.compensation == "DS-B1",:), 't_index');
    P.initial_20s_accuracy(m) = A.mean_accuracy(1);
    P.final_5y_accuracy(m) = A.mean_accuracy(end);
    P.max_drop_pp(m) = max(0, A.mean_accuracy(1) - min(A.mean_accuracy));
    if m == 2
        % Zero-based t_index 3,4,7,11 => MATLAB rows 4,5,8,12.
        delta = A.mean_accuracy([4,5,8,12]) - B.mean_accuracy([4,5,8,12]);
        P{m,9:12} = delta';
    end
end

fig = figure('Color','w', 'Name','Table III: drift robustness', ...
             'Position',[120,120,1600,300]);
ax = axes('Parent',fig, 'Position',[0.015,0.16,0.97,0.64]);
hold(ax,'on'); axis(ax,[0,1,0,1]); axis(ax,'off');
widths = [1.0,1.35,1.1,0.6,0.9,1.0,1.0,0.9,0.65,0.65,0.65,0.65];
edges = [0,cumsum(widths)/sum(widths)];
centers = (edges(1:end-1) + edges(2:end))/2;
headers = {'Model','Training','Compensation','lambda NSR', ...
    'Final Test','Initial 20s','Final 5Y','Max-drop','1h','1d','6M','5Y'};
for c = 1:4
    tableText(ax,centers(c),0.78,headers{c},'bold');
end
tableText(ax,mean(edges([5,9])),0.91,'Test Accuracy (%)','bold');
tableText(ax,mean(edges([9,13])),0.91,'Accuracy Improvement (pp)','bold');
for c = 5:12
    tableText(ax,centers(c),0.66,headers{c},'bold');
end
for y = [0.99,0.55,0.06]
    plot(ax,[0,1],[y,y],'-','Color','k','LineWidth',0.8);
end
plot(ax,edges([5,9]),[0.79,0.79],'-','Color','k','LineWidth',0.6);
plot(ax,edges([9,13]),[0.79,0.79],'-','Color','k','LineWidth',0.6);
for r = 1:2
    y = 0.43 - (r-1)*0.23;
    tableText(ax,centers(1),y,char(P.model(r)),'normal');
    tableText(ax,centers(2),y,char(P.training(r)),'normal');
    tableText(ax,centers(3),y,char(P.compensation(r)),'normal');
    if r == 1, coefficient = char(8212); else, coefficient = sprintf('%.1f',P.lambda_nsr(r)); end
    tableText(ax,centers(4),y,coefficient,'normal');
    for c = 5:12
        value = P{r,c};
        if isnan(value)
            label = char(8212);
        elseif c >= 9
            label = sprintf('%+.2f',value);
        else
            label = sprintf('%.2f',value);
        end
        weight = 'normal';
        if (c == 6 && value == max(P.initial_20s_accuracy)) || ...
           (c == 7 && value == max(P.final_5y_accuracy)) || ...
           (c == 8 && value == min(P.max_drop_pp))
            weight = 'bold';
        end
        tableText(ax,centers(c),y,label,weight);
    end
end
addText(fig,[0.02,0.94,0.96,0.05],'TABLE III',12);
caption = 'DRIFT ROBUSTNESS: HWA VERSUS NSR-HWA WITH THE SAME DRIFTSLICE B=1 COMPENSATION';
addText(fig,[0.02,0.83,0.96,0.10],caption,11);
addText(fig,[0.02,0.005,0.96,0.13], ...
    {'Max-drop = initial 20s accuracy minus the minimum across all 12 retention timestamps.', ...
     'Improvement = NSR-HWA minus HWA (percentage points). Final Test before PCM is unavailable in the supplied retention CSVs.'},10);
end


function tableText(ax,x,y,label,weight)
text(ax,x,y,label,'HorizontalAlignment','center','VerticalAlignment','middle', ...
    'FontName','Times New Roman','FontSize',11,'FontWeight',weight,'Interpreter','none');
end


function writeLatexTable(P, path)
% Use booktabs, multirow and graphicx packages in your manuscript preamble.
b = char(92);
lineEnd = [' ',b,b];
lines = { ...
    [b,'begin{table*}[t]']; ...
    [b,'centering']; ...
    [b,'caption{Drift robustness of HWA and NSR-HWA with identical DriftSlice B=1 compensation. ', ...
       'Accuracy is in percent; improvements are percentage points relative to HWA.}']; ...
    [b,'label{tab:nsr-drift-robustness}']; ...
    [b,'resizebox{',b,'textwidth}{!}{%']; ...
    [b,'begin{tabular}{llllrrrrrrrr}']; ...
    [b,'toprule']; ...
    ['Model & Training & Compensation & $',b,'lambda_{',b,'mathrm{NSR}}$ & ', ...
     b,'multicolumn{4}{c}{Test Accuracy (',b,'%)} & ', ...
     b,'multicolumn{4}{c}{Accuracy Improvement (pp)}',lineEnd]; ...
    [b,'cmidrule(lr){5-8}',b,'cmidrule(lr){9-12}']; ...
    [' & & & & Final Test & Initial 20s & Final 5Y & Max-drop & 1h & 1d & 6M & 5Y',lineEnd]; ...
    [b,'midrule']};
for r = 1:2
    values = cell(1,12);
    values(1:3) = {char(P.model(r)),char(P.training(r)),char(P.compensation(r))};
    if r == 1, values{4} = '--'; else, values{4} = sprintf('%.1f',P.lambda_nsr(r)); end
    for c = 5:12
        value = P{r,c};
        if isnan(value)
            values{c} = '--';
        elseif c >= 9
            values{c} = sprintf('%+.2f',value);
        else
            values{c} = sprintf('%.2f',value);
        end
        if (c == 6 && value == max(P.initial_20s_accuracy)) || ...
           (c == 7 && value == max(P.final_5y_accuracy)) || ...
           (c == 8 && value == min(P.max_drop_pp))
            values{c} = [b,'textbf{',values{c},'}'];
        end
    end
    lines{end+1} = [strjoin(values,' & '),lineEnd];
end
lines = [lines; { ...
    [b,'bottomrule']; [b,'end{tabular}}']; ...
    [b,'par',b,'smallskip',b,'footnotesize ', ...
     'Final Test before PCM is not available in the supplied retention CSVs. ', ...
     'Max-drop is initial 20s accuracy minus the minimum over the 12 retention times. ', ...
     'Values average three training repetitions, each with 30 device realizations.']; ...
    [b,'end{table*}']}];
fid = fopen(path,'w');
assert(fid >= 0,'Cannot create LaTeX table: %s',path);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',lines{:});
end
