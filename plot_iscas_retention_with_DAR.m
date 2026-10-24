function plot_iscas_retention_with_DAR(dataDir, outputDir)
%PLOT_ISCAS_RETENTION_WITH_DAR Plot verified HWA, DAR-HWA and NSR-HWA results.
% MATLAB R2020a or newer; no additional toolboxes required.
%
% Unzip the supplied package, open that folder in MATLAB, then run:
%   plot_iscas_retention_with_DAR
%
% Optional explicit folders:
%   plot_iscas_retention_with_DAR('C:\my_data', 'C:\my_figures')
%
% Input: source_fold_means.csv (already averaged over 30 realizations/fold).
% Error = 100 - accuracy; bands = mean +/- 2 SAMPLE SD across fold means.
% Table III reports all three training methods and improvement over HWA + GDC-safe.
% The x-axis uses equally spaced timestamp positions, as in the screenshots.
% No smoothing, curve fitting, training, or parameter tuning is performed.

if nargin < 1 || isempty(dataDir)
    dataDir = fileparts(mfilename('fullpath'));
end
if nargin < 2 || isempty(outputDir)
    outputDir = fullfile(dataDir, 'matlab_figures_with_DAR');
end
inputFile = fullfile(dataDir, 'source_fold_means.csv');
assert(isfile(inputFile), 'Missing input CSV: %s', inputFile);
T = readtable(inputFile, 'TextType', 'string');
required = {'method', 'compensation', 't_index', 't_s', 'fold', 'accuracy'};
assert(all(ismember(required, T.Properties.VariableNames)), ...
    'The input CSV is missing required columns.');
T.method = string(T.method);
T.compensation = string(T.compensation);

methods = ["PCM-HWA", "PCM-DAR-HWA", "PCM-NSR-HWA"];
compensations = ["none", "GDC-safe", "DS-B1", "DS-val"];
T = T(ismember(T.method, methods) & ismember(T.compensation, compensations), :);
assert(~isempty(T), 'No HWA/DAR/NSR data found.');
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
assert(F == 3, 'This package is the supplied three-repetition full run.');

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
purple = [136,78,160]/255;

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
title(ax1, '(a) Training comparison: DriftSlice B=1', 'FontSize',14, 'Interpreter','none');
title(ax2, '(b) Compensation comparison: PCM-HWA', 'FontSize',14, 'Interpreter','none');

h1 = drawCurve(ax1, S, "PCM-HWA", "DS-B1", black);
h2 = drawCurve(ax1, S, "PCM-NSR-HWA", "DS-B1", orange);
hDAR = drawCurve(ax1, S, "PCM-DAR-HWA", "DS-B1", purple);
uistack([h1,hDAR,h2], 'top');
legend(ax1, [h1,hDAR,h2], {'HWA + DriftSlice B=1','DAR-HWA + DriftSlice B=1','NSR-HWA + DriftSlice B=1'}, ...
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
H = table('Size',[14,7], ...
    'VariableTypes',{'string','string','string','string','string','double','double'}, ...
    'VariableNames',{'panel','candidate_method','candidate_compensation', ...
                     'control_method','control_compensation', ...
                     'delta_accuracy_20s_pp','delta_accuracy_5y_pp'});
leftInitial = zeros(2,4); leftFinal = zeros(2,4);
rightInitial = zeros(3,2); rightFinal = zeros(3,2);
k = 0;
for m = 2:3
    for c = 1:4
        [a,b] = endpointDifference(S, methods(m), compensations(c), "PCM-HWA", compensations(c));
        leftInitial(m-1,c) = a; leftFinal(m-1,c) = b;
        k = k + 1;
        H(k,:) = {"Training minus HWA (same compensation)", methods(m), ...
            compensations(c), "PCM-HWA", compensations(c), a, b};
    end
end
for m = 1:3
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
              'Position',[100,100,1600,700]);
ax3 = axes('Parent',fig2, 'Position',[0.08,0.29,0.50,0.50]);
ax4 = axes('Parent',fig2, 'Position',[0.67,0.17,0.22,0.66]);
colormap(fig2, map);
drawTriangles(ax3, leftInitial, leftFinal, limit, ...
    {'DAR - HWA','NSR - HWA'}, {'None','GDC-safe','DS-B1','DS-val'});
drawTriangles(ax4, rightInitial, rightFinal, limit, ...
    {'HWA','DAR-HWA','NSR-HWA'}, {'DS-B1','DS-val'});
title(ax3, 'Training minus HWA (same compensation)', 'FontSize',14, 'Interpreter','none');
title(ax4, 'DriftSlice minus GDC-safe (same training)', 'FontSize',14, 'Interpreter','none');
cb = colorbar(ax4);
set(ax4, 'Position',[0.67,0.17,0.22,0.66]);
set(cb, 'Position',[0.922,0.17,0.012,0.66], 'FontSize',12);
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
saveFigure(fig1, outputDir, 'figure1_test_error_with_DAR');
saveFigure(fig2, outputDir, 'figure2_split_triangle_heatmap_with_DAR');
[fig3, paperTable] = drawRetentionTable(S);
writetable(paperTable, fullfile(outputDir, 'table_III_all_training_vs_hwa_gdc.csv'));
writeLatexTable(paperTable, fullfile(outputDir, 'table_III_all_training_vs_hwa_gdc.tex'));
saveFigure(fig3, outputDir, 'table_III_all_training_vs_hwa_gdc');
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
methods = ["PCM-HWA","PCM-DAR-HWA","PCM-NSR-HWA"];
trainings = ["HWA","DAR-HWA","NSR-HWA"];
comps = ["GDC-safe","DS-B1","DS-val"];
P = table('Size',[9,11], ...
    'VariableTypes',{'string','string','string','double','double','double','double', ...
                     'double','double','double','double'}, ...
    'VariableNames',{'model','training','compensation','final_test_accuracy', ...
        'initial_20s_accuracy','final_5y_accuracy','max_drop_pp','improvement_1h_pp', ...
        'improvement_1d_pp','improvement_6mo_pp','improvement_5y_pp'});
baseline = sortrows(S(S.method == "PCM-HWA" & S.compensation == "GDC-safe",:),'t_index');
k = 0;
for m = 1:3
    for c = 1:3
        k = k+1;
        A = sortrows(S(S.method == methods(m) & S.compensation == comps(c),:),'t_index');
        delta = (A.mean_accuracy([4,5,8,12])-baseline.mean_accuracy([4,5,8,12]))';
        if k == 1, delta(:) = NaN; end
        P(k,:) = {"ResNet-32",trainings(m),comps(c),NaN,A.mean_accuracy(1), ...
            A.mean_accuracy(end),A.mean_accuracy(1)-min(A.mean_accuracy), ...
            delta(1),delta(2),delta(3),delta(4)};
    end
end
fig = figure('Color','w','Name','Table III: HWA, DAR-HWA and NSR-HWA', ...
    'Position',[120,120,1650,640]);
ax = axes('Parent',fig,'Position',[0.015,0.18,0.97,0.64]);
hold(ax,'on'); axis(ax,[0,1,0,1]); axis(ax,'off');
widths = [1,1.0,1.8,0.85,1,1,0.9,0.65,0.65,0.65,0.65];
edges = [0,cumsum(widths)/sum(widths)];
centers = (edges(1:end-1)+edges(2:end))/2;
headers = {'Model','Training','Compensation','Final Test','Initial 20s', ...
    'Final 5Y','Max-drop','1h','1d','6M','5Y'};
for c = 1:3, tableText(ax,centers(c),0.90,headers{c},'bold'); end
tableText(ax,mean(edges([4,8])),0.96,'Test Accuracy (%)','bold');
tableText(ax,mean(edges([8,12])),0.96,'Gain vs HWA + GDC-safe (pp)','bold');
for c = 4:11, tableText(ax,centers(c),0.85,headers{c},'bold'); end
for y = [1,0.79,0.035], plot(ax,[0,1],[y,y],'-','Color','k','LineWidth',0.8); end
plot(ax,edges([4,8]),[0.91,0.91],'-','Color','k','LineWidth',0.6);
plot(ax,edges([8,12]),[0.91,0.91],'-','Color','k','LineWidth',0.6);
for r = 1:9
    y = 0.73-(r-1)*0.08;
    tableText(ax,centers(1),y,char(P.model(r)),'normal');
    tableText(ax,centers(2),y,char(P.training(r)),'normal');
    tableText(ax,centers(3),y,compensationLabel(P.compensation(r),r == 1),'normal');
    for c = 4:11
        v = P{r,c};
        if isnan(v), label = char(8212);
        elseif c >= 8, label = sprintf('%+.2f',v);
        else, label = sprintf('%.2f',v);
        end
        weight = 'normal';
        if (c == 5 && v == max(P.initial_20s_accuracy)) || ...
           (c == 6 && v == max(P.final_5y_accuracy)) || ...
           (c == 7 && v == min(P.max_drop_pp)), weight = 'bold'; end
        tableText(ax,centers(c),y,label,weight);
    end
    if r == 3 || r == 6
        plot(ax,[0,1],[y-0.04,y-0.04],'-','Color',[0.6,0.6,0.6],'LineWidth',0.5);
    end
end
addText(fig,[0.02,0.95,0.96,0.04],'TABLE III',12);
addText(fig,[0.02,0.85,0.96,0.09], ...
    'HWA + GDC-SAFE BASELINE AND DRIFTSLICE UNDER HWA, DAR-HWA AND NSR-HWA TRAINING',12);
addText(fig,[0.02,0.005,0.96,0.14], ...
    {'All gains use HWA + GDC-safe as the baseline. Means across 3 repetitions, each with 30 device realizations.', ...
     'Training and compensation vary across rows; matched-training comparisons isolate compensation.', ...
     'Max-drop = initial 20s accuracy minus the minimum over 12 timestamps. Final Test before PCM is unavailable.'},10);
end


function tableText(ax,x,y,label,weight)
text(ax,x,y,label,'HorizontalAlignment','center','VerticalAlignment','middle', ...
    'FontName','Times New Roman','FontSize',11,'FontWeight',weight,'Interpreter','none');
end


function label = compensationLabel(comp,isBaseline)
if comp == "DS-B1", label = 'DriftSlice B=1';
elseif comp == "DS-val", label = 'DriftSlice selected B';
elseif isBaseline, label = 'GDC-safe (baseline)';
else, label = 'GDC-safe';
end
end


function writeLatexTable(P,path)
b = char(92); lineEnd = [' ',b,b];
lines = { ...
    [b,'begin{table*}[t]']; [b,'centering']; ...
    [b,'caption{HWA, DAR-HWA and NSR-HWA retention results. ', ...
       'All improvements are relative to HWA + GDC-safe. Accuracy is in percent; gains are percentage points.}']; ...
    [b,'label{tab:all-training-driftslice}']; ...
    [b,'resizebox{',b,'textwidth}{!}{%']; ...
    [b,'begin{tabular}{lllrrrrrrrr}']; [b,'toprule']; ...
    ['Model & Training & Compensation & ',b,'multicolumn{4}{c}{Test Accuracy (',b,'%)} & ', ...
     b,'multicolumn{4}{c}{Gain vs HWA + GDC-safe (pp)}',lineEnd]; ...
    [b,'cmidrule(lr){4-7}',b,'cmidrule(lr){8-11}']; ...
    [' & & & Final Test & Initial 20s & Final 5Y & Max-drop & 1h & 1d & 6M & 5Y',lineEnd]; ...
    [b,'midrule']};
for r = 1:height(P)
    values = cell(1,11);
    values(1:3) = {char(P.model(r)),char(P.training(r)),compensationLabel(P.compensation(r),r == 1)};
    for c = 4:11
        v = P{r,c};
        if isnan(v), values{c} = '--';
        elseif c >= 8, values{c} = sprintf('%+.2f',v);
        else, values{c} = sprintf('%.2f',v);
        end
        if (c == 5 && v == max(P.initial_20s_accuracy)) || ...
           (c == 6 && v == max(P.final_5y_accuracy)) || ...
           (c == 7 && v == min(P.max_drop_pp))
            values{c} = [b,'textbf{',values{c},'}'];
        end
    end
    lines{end+1} = [strjoin(values,' & '),lineEnd];
    if r == 3 || r == 6, lines{end+1} = [b,'midrule']; end
end
lines = [lines; { ...
    [b,'bottomrule']; [b,'end{tabular}}']; ...
    [b,'par',b,'smallskip',b,'footnotesize Means across three training repetitions, each with 30 device realizations. ', ...
     'Training and compensation vary across rows; matched-training comparisons isolate compensation. ', ...
     'Max-drop is initial 20s accuracy minus the minimum over 12 retention times. ', ...
     'Final Test before PCM is unavailable in the retention CSVs.']; ...
    [b,'end{table*}']}];
fid = fopen(path,'w'); assert(fid >= 0,'Cannot create LaTeX table: %s',path);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',lines{:});
end


