% This is a demo demonstrating how to calculate and display the progression
% pattern of each brain region after determining the AD age for each subject.
% It only shows images five years after the onset of dementia; 
% the displayed time can be changed by altering the time variable 't'.
% ADage can be obtained from the result calculated by computeNonParaEMFixed.
t = 5; % five years after the onset of dementia
[ppcellROI_mu,cell_e_hc,cell_e_pt,cell_thetaMu,cell_beta,cell_ydif,monoROI,stdYROI] = ...
    computeROICurves(ROIptTable,ROIhcTable,isMRIROI,Xcolnames,DemoNames,ADage,knots,lambda);
minz = -0.5;
maxz = 1;
[MRIROIvalues,FBBROIvalues,AV45ROIvalues,AV1451ROIvalues] = ...
    computeADNITimeROIZvalue(ppcellROI_mu,cell_e_hc,ROIhcTable,newLabels,Xcolnames,t);
drawTimeSurface(MRIROIvalues,CData,fv,mymap,minz,maxz,'MRI_5.tif');
drawTimeSurface(AV45ROIvalues,CData,fv,mymap,minz,maxz,'AV45_5.tif');
drawTimeSurface(FBBROIvalues,CData,fv,mymap,minz,maxz,'FBB_5.tif');
drawTimeSurface(AV1451ROIvalues,CData,fv,mymap,minz,maxz,'AV1451_5.tif');



