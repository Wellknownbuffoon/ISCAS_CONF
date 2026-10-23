function draw_iscas_table_III(dataDir, outputDir)
%DRAW_ISCAS_TABLE_III Draw only the requested table from the supplied CSVs.
% MATLAB R2020a or newer; no additional toolboxes required.
% Unzip the table-only package, open that folder, run: draw_iscas_table_III
% Input data are your completed ResNet-32 full run, not the reference paper.
if nargin < 1 || isempty(dataDir)
    dataDir = fileparts(mfilename('fullpath'));
end
if nargin < 2 || isempty(outputDir)
    outputDir = fullfile(dataDir,'matlab_table');
end
inputFile = fullfile(dataDir,'source_summary.csv');
assert(isfile(inputFile),'Keep source_summary.csv beside this MATLAB file.');
S = readtable(inputFile,'TextType','string');
required = {'method','compensation','t_index','mean_accuracy','n_folds'};
assert(all(ismember(required,S.Properties.VariableNames)), 'Input CSV columns differ.');
S.method = string(S.method);
S.compensation = string(S.compensation);
for method = ["PCM-HWA","PCM-NSR-HWA"]
    A = sortrows(S(S.method == method & S.compensation == "DS-B1",:),'t_index');
    assert(height(A) == 12 && isequal(A.t_index,(0:11)'), 'Missing or duplicated timestamps.');
    assert(all(isfinite(A.mean_accuracy) & A.mean_accuracy >= 0 & A.mean_accuracy <= 100), ...
        'Invalid accuracy values.');
    assert(all(A.n_folds == 3), 'This standalone table is for the supplied three-repetition full run.');
end
if ~isfolder(outputDir), mkdir(outputDir); end
[fig, P] = drawRetentionTable(S);
writetable(P,fullfile(outputDir,'table_III_drift_robustness.csv'));
writeLatexTable(P,fullfile(outputDir,'table_III_drift_robustness.tex'));
saveFigure(fig,outputDir,'table_III_drift_robustness');
disp(P);
fprintf('Saved table PNG, PDF, FIG, CSV and LaTeX in:\n%s\n',outputDir);
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
