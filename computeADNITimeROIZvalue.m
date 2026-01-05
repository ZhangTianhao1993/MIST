function [MRIROIvalues,FBBROIvalues,AV45ROIvalues,AV1451ROIvalues] = ...
    computeADNITimeROIZvalue(ppcellROI_mu,cell_ecn,ROINegCNTable,newLabels,Xcolnames,timepoint)
    n = length(ppcellROI_mu);
    ROIvalues = cell(n,3);
    zvalue = NaN(n,1);
    for i=1:n
        mui = fnval(ppcellROI_mu{i},timepoint);
        e_cn = cell_ecn{i};
        AGE = ROINegCNTable.AGE;
        zvalue(i) = mui/mean(ROINegCNTable(:,Xcolnames{i}).Variables,'omitnan');
        ROIvalues(i,1) = Xcolnames(i);
        if mod(i,84) == 0
            ni = 84;
        else
            ni = mod(i,84);
        end
        ROIvalues(i,2) = {cell2mat(newLabels(ni,2))};
        ROIvalues(i,3) = {zvalue(i)};
    end
    MRIROIvalues = ROIvalues(1:84,:);
    FBBROIvalues = ROIvalues(85:168,:);
    AV45ROIvalues = ROIvalues(169:252,:);
    AV1451ROIvalues = ROIvalues(253:336,:);
end
