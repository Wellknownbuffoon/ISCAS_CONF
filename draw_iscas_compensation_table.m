function draw_iscas_compensation_table(dataDir, outputDir)
%DRAW_ISCAS_COMPENSATION_TABLE HWA + GDC-safe versus HWA + DriftSlice.
% Unzip the supplied package, open its folder in MATLAB R2020a+, and run:
%   draw_iscas_compensation_table
% Both baseline and proposed rows use the SAME PCM-HWA trained checkpoints.
% No NSR training comparison is made in this table. No toolboxes are needed.
if nargin < 1 || isempty(dataDir)
    dataDir = fileparts(mfilename('fullpath'));
end
if nargin < 2 || isempty(outputDir)
    outputDir = fullfile(dataDir,'compensation_table');
end
path = fullfile(dataDir,'source_fold_means.csv');
assert(isfile(path),'Keep source_fold_means.csv beside this MATLAB file.');
T = readtable(path,'TextType','string');
required = {'method','compensation','fold','t_index','t_s','accuracy'};
assert(all(ismember(required,T.Properties.VariableNames)),'Input CSV columns differ.');
T.method = string(T.method); T.compensation = string(T.compensation);
comps = ["GDC-safe"; "DS-B1"; "DS-val"];
T = T(T.method == "PCM-HWA" & ismember(T.compensation,comps),:);
assert(~isempty(T),'No PCM-HWA compensation results found.');
assert(height(unique(T(:,{'compensation','fold','t_index'}))) == height(T), ...
    'Duplicate fold-mean records.');
folds = unique(T.fold); F = numel(folds);
assert(F == 3,'This package contains the supplied three-repetition full run.');
expectedTimes = [20;60;1800;3600;86400;432000;2592000;15552000; ...
                 31557600;63115200;94672800;157788000];
accuracy = zeros(12,3);
for c = 1:3
    for ti = 0:11
        A = T(T.compensation == comps(c) & T.t_index == ti,:);
        assert(height(A) == F && isequal(unique(A.fold),folds), ...
            'Missing or unmatched folds for %s, timestamp %d.',char(comps(c)),ti);
        assert(all(abs(A.t_s-expectedTimes(ti+1)) < 1e-6),'Retention time mismatch.');
        assert(all(isfinite(A.accuracy) & A.accuracy >= 0 & A.accuracy <= 100), ...
            'Accuracy must be finite and expressed in percent.');
        accuracy(ti+1,c) = mean(A.accuracy);
    end
end

P = table(repmat("ResNet-32",3,1),repmat("HWA",3,1),comps, ...
    nan(3,1),accuracy(1,:)',accuracy(end,:)', ...
    (accuracy(1,:)-min(accuracy,[],1))',nan(3,1),nan(3,1),nan(3,1),nan(3,1), ...
    'VariableNames',{'model','training','compensation','final_test_accuracy', ...
    'initial_20s_accuracy','final_5y_accuracy','max_drop_pp','improvement_1h_pp', ...
    'improvement_1d_pp','improvement_6mo_pp','improvement_5y_pp'});
for c = 2:3
    % Improvement = proposed accuracy minus the HWA + GDC-safe baseline.
    P{c,8:11} = (accuracy([4,5,8,12],c)-accuracy([4,5,8,12],1))';
end

fig = figure('Color','w','Name','Table III: GDC-safe versus DriftSlice', ...
    'Position',[100,100,1650,350]);
ax = axes('Parent',fig,'Position',[0.015,0.19,0.97,0.62]);
hold(ax,'on'); axis(ax,[0,1,0,1]); axis(ax,'off');
widths = [1,0.7,2.2,0.9,1.0,1.0,0.95,0.65,0.65,0.65,0.65];
edges = [0,cumsum(widths)/sum(widths)];
centers = (edges(1:end-1)+edges(2:end))/2;
headers = {'Model','Training','Compensation','Final Test','Initial 20s', ...
    'Final 5Y','Max-drop','1h','1d','6M','5Y'};
for c = 1:3
    tableText(ax,centers(c),0.80,headers{c},'bold');
end
tableText(ax,mean(edges([4,8])),0.92,'Test Accuracy (%)','bold');
tableText(ax,mean(edges([8,12])),0.92,'Improvement over GDC-safe (pp)','bold');
for c = 4:11
    tableText(ax,centers(c),0.70,headers{c},'bold');
end
for y = [1,0.61,0.02]
    plot(ax,[0,1],[y,y],'-','Color','k','LineWidth',0.8);
end
plot(ax,edges([4,8]),[0.82,0.82],'-','Color','k','LineWidth',0.6);
plot(ax,edges([8,12]),[0.82,0.82],'-','Color','k','LineWidth',0.6);
names = {'GDC-safe (baseline)','DriftSlice B=1 (proposed)', ...
         'DriftSlice selected B (proposed)'};
for r = 1:3
    y = 0.50-(r-1)*0.19;
    tableText(ax,centers(1),y,char(P.model(r)),'normal');
    tableText(ax,centers(2),y,char(P.training(r)),'normal');
    tableText(ax,centers(3),y,names{r},'normal');
    for c = 4:11
        v = P{r,c};
        if isnan(v), label = char(8212);
        elseif c >= 8, label = sprintf('%+.2f',v);
        else, label = sprintf('%.2f',v);
        end
        weight = 'normal';
        if (c == 5 && v == max(P.initial_20s_accuracy)) || ...
           (c == 6 && v == max(P.final_5y_accuracy)) || ...
           (c == 7 && v == min(P.max_drop_pp))
            weight = 'bold';
        end
        tableText(ax,centers(c),y,label,weight);
    end
end
addText(fig,[0.02,0.94,0.96,0.05],'TABLE III',12);
addText(fig,[0.02,0.83,0.96,0.10], ...
    'BASELINE HWA + GDC-SAFE VERSUS PROPOSED HWA + DRIFTSLICE',12);
addText(fig,[0.02,0.005,0.96,0.15], ...
    {'Same HWA-trained checkpoints; means across 3 training repetitions, each with 30 device realizations.', ...
     'Max-drop = initial 20s accuracy minus the minimum over 12 retention timestamps. Selected B uses validation data.', ...
     'Improvement = DriftSlice minus GDC-safe (percentage points). Final Test before PCM is unavailable.'},10);

if ~isfolder(outputDir), mkdir(outputDir); end
stem = fullfile(outputDir,'table_III_hwa_gdc_vs_driftslice');
writetable(P,[stem,'.csv']);
writeLatexTable(P,[stem,'.tex']);
drawnow;
exportgraphics(fig,[stem,'.png'],'Resolution',300,'BackgroundColor','white');
exportgraphics(fig,[stem,'.pdf'],'ContentType','vector','BackgroundColor','white');
savefig(fig,[stem,'.fig']);
disp(P);
fprintf('Saved corrected table PNG, PDF, FIG, CSV and LaTeX in:\n%s\n',outputDir);
end


function tableText(ax,x,y,label,weight)
text(ax,x,y,label,'HorizontalAlignment','center','VerticalAlignment','middle', ...
    'FontName','Times New Roman','FontSize',11,'FontWeight',weight,'Interpreter','none');
end


function addText(fig,position,value,fontSize)
annotation(fig,'textbox',position,'String',value,'EdgeColor','none', ...
    'HorizontalAlignment','center','VerticalAlignment','middle', ...
    'FontName','Arial','FontSize',fontSize,'Interpreter','none');
end


function writeLatexTable(P,path)
b = char(92); lineEnd = [' ',b,b];
lines = { ...
    [b,'begin{table*}[t]']; [b,'centering']; ...
    [b,'caption{Drift robustness of baseline HWA + GDC-safe versus proposed HWA + DriftSlice. ', ...
       'The HWA-trained checkpoints are held fixed. Accuracy is in percent; improvements are percentage points.}']; ...
    [b,'label{tab:hwa-gdc-driftslice}']; ...
    [b,'resizebox{',b,'textwidth}{!}{%']; ...
    [b,'begin{tabular}{lllrrrrrrrr}']; [b,'toprule']; ...
    ['Model & Training & Compensation & ',b,'multicolumn{4}{c}{Test Accuracy (',b,'%)} & ', ...
     b,'multicolumn{4}{c}{Improvement over GDC-safe (pp)}',lineEnd]; ...
    [b,'cmidrule(lr){4-7}',b,'cmidrule(lr){8-11}']; ...
    [' & & & Final Test & Initial 20s & Final 5Y & Max-drop & 1h & 1d & 6M & 5Y',lineEnd]; ...
    [b,'midrule']};
names = {'GDC-safe (baseline)','DriftSlice B=1 (proposed)','DriftSlice selected B (proposed)'};
for r = 1:3
    values = cell(1,11);
    values(1:3) = {char(P.model(r)),char(P.training(r)),names{r}};
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
end
lines = [lines; { ...
    [b,'bottomrule']; [b,'end{tabular}}']; ...
    [b,'par',b,'smallskip',b,'footnotesize Same HWA-trained checkpoints; ', ...
     'means across three training repetitions, each with 30 device realizations. ', ...
     'Max-drop is initial 20s accuracy minus the minimum over all 12 retention times. ', ...
     'Selected B uses validation data. Final Test before PCM is unavailable in the retention CSVs.']; ...
    [b,'end{table*}']}];
fid = fopen(path,'w');
assert(fid >= 0,'Cannot create LaTeX table: %s',path);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s\n',lines{:});
end
