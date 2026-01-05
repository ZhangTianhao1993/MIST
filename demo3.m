% This is a demo demonstrating how to predict AD age using only cross-sectional data
% What is displayed here is the use of MRI features combined with cognition for prediction
% XthetaMu_madel, Xbeta_madel, and t_cn_cci_madel are the estimated curve parameters, 
% demographic coefficients, and the boundary between CN and MCI based on the ADNI database.
MRIcogfeatureNum = [1:84,253:256];
preTable_MRIcog = UsingMultiFeaturesPredictADage(MRIfeatureTable,...
    [97:97+83,269,268,271,273],[{'AGE'},{'GENDER'},{'EDUC'},{'E4'},{'E2'},{'ICV_Harm'}],...
    "DIAGFULL",knots_model,XthetaMu_model(:,MRIcogfeatureNum),...
    Xbeta_model(:,MRIcogfeatureNum),t_cn_mci_model,rho_model,...
[1/84*ones(1,84),1/4*ones(1,4)],XstdY_model(MRIcogfeatureNum));
x = preTable_MRIcog(preTable_MRIcog.DIAGFULL == 2,:).ADage;
y = preTable_MRIcog(preTable_MRIcog.DIAGFULL == 2,:).predictADage;
indx = isnan(y);
x(indx) = []; y(indx) = [];
figure;
hold on
scatter(x,y);
[p,S] = polyfit(x,y,1);
minx = min(x)-0.05*(max(x)-min(x));
maxx = max(x)+0.05*(max(x)-min(x));
X = [minx,maxx];
Y = [p(1)*minx+p(2),p(1)*maxx+p(2)];
plot(X,Y);
hold off
