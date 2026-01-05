function [ppcellROI_mu,cell_e_hc,cell_e_pt,cell_thetaMu,cell_beta,cell_ydif,monoROI,stdYROI] = ...
    computeROICurves(ROIptTable,ROIhcTable,isMRIROI,Xcolnames,DemoNames,ADage,knots,lambda)
%--------------------------------------------------------------------------
% computeROICurves
%
% ROI-level biomarker trajectory estimation with fixed disease age.
%
% This function estimates region-of-interest (ROI)–specific biomarker
% progression curves along a predefined Alzheimer’s disease (AD) timeline.
% Unlike the full EM framework, subject-specific disease ages (ADage) are
% assumed to be known and fixed. The function is primarily intended for
% post hoc analysis, visualization, and region-wise characterization of
% biomarker dynamics.
%
% Methodological context:
%   Given disease ages estimated from the global non-parametric EM model,
%   this function fits monotonic cubic B-spline trajectories independently
%   for each ROI. Demographic effects are regressed out, and smoothness is
%   enforced through a second-derivative penalty. The procedure mirrors the
%   M-step of the full model while omitting latent onset estimation.
%
% Inputs:
%   ROIptTable - Table containing patients ROI-level data.
%   ROIhcTable  - Table containing hc subjects ROI-level data.
%   isMRIROI       - Logical vector indicating MRI-derived ROIs.
%   Xcolnames      - Variable names of ROI biomarkers. (Cell)
%   DemoNames      - Variable names for demographic covariates.
%   ADage          - Fixed disease age for each visit.
%   knots          - Knot locations for cubic B-spline basis.
%   lambda         - Smoothness regularization parameter.
%
% Outputs:
%   ppcellROI_mu   - Cell array of fitted ROI trajectory spline functions.
%   cell_e_hc       - Cell array of hc residuals after demographic correction.
%   cell_e_pt       - Cell array of pt residuals after trajectory subtraction.
%   cell_thetaMu   - Cell array of estimated spline coefficients per ROI.
%   cell_beta      - Cell array of demographic regression coefficients.
%   cell_ydif      - Cell array of demographically adjusted pt ROI signals.
%   monoROI        - Vector indicating enforced monotonic direction per ROI.
%
% Notes:
%   - Monotonicity is enforced via linear inequality constraints on spline
%     coefficients.
%   - MRI-derived ROIs can be constrained to have zero baseline offset and
%     slope, ensuring biologically plausible trajectories.
%   - Each ROI is modeled independently, enabling region-wise inspection
%     and visualization of disease progression patterns.
%
% Author:
%   Tianhao Zhang (Zhang Tianhao)
%
% Date:
%   January 05, 2026
%--------------------------------------------------------------------------

D_ad = ROIptTable(:,DemoNames).Variables;
D_cn = ROIhcTable(:,DemoNames).Variables;
D_ad = [ones(size(ROIptTable,1),1),D_ad];
D_cn = [ones(size(ROIhcTable,1),1),D_cn];
D_ad(isnan(D_ad)) = 0;
D_cn(isnan(D_cn)) = 0;
n = length(Xcolnames);
ppcellROI_mu = cell(n,1);
cell_e_hc = cell(n,1);
cell_e_pt = cell(n,1);
cell_thetaMu = cell(n,1);
cell_beta = cell(n,1);

Y_ad = ROIptTable(:,Xcolnames).Variables;
Y_cn = ROIhcTable(:,Xcolnames).Variables;
r = mean(Y_ad,'omitnan') - mean(Y_cn,'omitnan') ;
monoROI = 2*(r>0)-1;
monoROI = 1-2*double(isMRIROI');
Y_ad = Y_ad.*monoROI;
Y_cn = Y_cn.*monoROI;
Y = [Y_cn;Y_ad];
stdYROI = std(Y,'omitnan');
Y_ad = (Y_ad)./stdYROI;
Y_cn = (Y_cn)./stdYROI; 
omega = BaseFunMatrix(knots);
pointNum = length(knots);
Dfilled = fillmissing(D_cn,'nearest');
Yfilled = fillmissing(Y_cn,"nearest");
beta = Dfilled\Yfilled;
beta(end,isMRIROI==0) = 0;
options1 = optimoptions(@fmincon,'Display','none','MaxIterations',500,...
    'MaxFunctionEvaluations',10^9,'UseParallel',false,...
    'OptimalityTolerance',1e-5,'StepTolerance',1e-10,...
    'Algorithm','sqp','FiniteDifferenceType','forward');%'Algorithm','sqp',
parfor i=1:n
    %Xcolnames(i) = ROIADLikeTable(:,Xcolnames(i)).Properties.VariableNames;
    betai = beta(:,i);
    
    % thetaMui = thetaMu(:,i);
    % thetaMui(1:3) = 0;
    y_ad_i = Y_ad(:,i);
    y_cn_i = Y_cn(:,i);
    [~,thetaMui] = csaps_knots(ADage,y_ad_i-D_ad*betai,knots,lambda);
    thetaMui([1,2,3]) = 0;
    thetaMui(thetaMui<0) = 0;
    %thetaMui = sort(thetaMui);
    xi0 = [thetaMui(:);betai(:)];
    isMRIi = isMRIROI(i);
    xi= fmincon(@(x)objfunFixAgeonset(x,y_ad_i,y_cn_i,D_ad,D_cn,ADage,knots,omega,lambda),...
    xi0,[],[],[],[],[],[],@(x)myconFixAgeonset(x,pointNum,D_ad,isMRIi),options1);
    betai = xi(pointNum+3:end)*monoROI(i)*stdYROI(i);
    thetaMui = xi(1:pointNum+2)*monoROI(i)*stdYROI(i);
    %thetaMu(:,i) = thetaMui;
    %beta(:,i) = betai;
    sp_mu = spmak(augknt(knots,4),thetaMui');
    pp_mu = fnxtr(sp_mu,2);
    fxmu = fnval(pp_mu,ADage);
    ppcellROI_mu{i} = pp_mu;
    cell_e_hc{i} = Y_cn(:,i)*monoROI(i)*stdYROI(i) - D_cn*betai;
    cell_e_pt{i} = Y_ad(:,i)*monoROI(i)*stdYROI(i) - D_ad*betai-fxmu;
    cell_ydif{i} = Y_ad(:,i)*monoROI(i)*stdYROI(i) - D_ad*betai;
    cell_thetaMu{i} = thetaMui;
    cell_beta{i} = betai;
    %fprintf('%d\n',i);
end
end
function loss = objfunFixAgeonset(x,y_ad_i,y_cn_i,D_ad,D_cn,ADage,knots,omega,lambda)
    pointNum = length(knots); 
    [adNum,demoNum] = size(D_ad); 

    p1 = pointNum+2;
    thetaMu = x(1:p1);
    p2 = p1+demoNum;
    beta = x(p1+1:p2);
    
    t1 = (y_cn_i - D_cn*beta).^2;
    fxi_mu = fnxtr(spmak(augknt(knots,4),thetaMu'),2);
    fxmu = fnval(fxi_mu,ADage);
    tempvar1 = thetaMu*thetaMu';
    smoothing_mu = sum(tempvar1.*omega,"all");
    t2 = (y_ad_i - D_ad*beta - fxmu).^2;
    t2 = [t2;lambda*smoothing_mu];
    loss_cn = mean(t1,'omitnan');
    loss_pt = mean(t2,'omitnan');
    loss = loss_cn+loss_pt;
end
function [c,ceq] = myconFixAgeonset(x,pointNum,D_ad,isMRIi)
    [adNum,demoNum] = size(D_ad); 
    p1 = pointNum+2;
    thetaMu = x(1:p1);
    p2 = p1+demoNum;
    beta = x(p1+1:p2);
    A = diag(ones(pointNum+2,1)) + diag(-ones(pointNum+1,1),1);
    A(pointNum+2,:) = [];
    c = A*thetaMu;
    if isMRIi == 0
        ceq = [thetaMu(1);thetaMu(2);thetaMu(3);beta(end)];
    else
        ceq = [thetaMu(1);thetaMu(2);thetaMu(3)];
    end
end
function omega = BaseFunMatrix(knots)
    thetaNum = length(knots)+2;
    knots = augknt(knots,4);
    NFun = cell(thetaNum,1);
    d2NFun = cell(thetaNum,1);
    omega = NaN(thetaNum,thetaNum);
    for i=1:thetaNum
        NFun{i} = fnxtr(bspline(knots(i:i+4)),0);
        d2NFun{i} = fnder(NFun{i},2);
    end
    for i=1:thetaNum
        for j=1:thetaNum
            if abs(i-j)<4
                ddfx = fncmb(d2NFun{i},'*',d2NFun{j});
                omega(i,j) = diff(fnval(fnint(ddfx),[ddfx.breaks(1),ddfx.breaks(end)]));
            else
                omega(i,j) = 0;
            end
        end
    end
end
function [pp,theta] = csaps_knots(x,y,knots,lambda)
w = ones(length(x),1);

knots = knots(:)';

existIndx = ~(isnan(x)|isnan(y)|isnan(w));
x = x(existIndx);
y = y(existIndx);
w = w(existIndx);

[x,si] = sort(x);
y = y(si);
w = w(si);

x = reshape(x,[1,length(x)]);

outknotsIndx = x>max(knots) | x<min(knots);
x(outknotsIndx) = [];
y(outknotsIndx) = [];
w(outknotsIndx) = [];

[ux,~,ix] = unique(x);
new_w = accumarray(ix,w);

W = diag(new_w);

y = reshape(y,[length(y),1]);

y = accumarray(ix,w.*y)./accumarray(ix,w);
[N,Omega,ppcell] = BasFunMatrix(ux,knots);
theta = (transpose(N)*W*N + lambda*Omega)\(transpose(N)*W*y);
sp = spmak(augknt(knots,4),theta');
pp = fn2fm(sp,'pp');
end

function [N,Omega,ppcell] = BasFunMatrix(x,knots)
x = sort(x);
x = reshape(x,[1,length(x)]);

m = length(x);
n = length(knots);

knots = augknt(knots,4);


ppcell = cell(n+2,1);
spcell = cell(n+2,1);

ddppcell = cell(n+2,1);
for j=1:n+2
    ppcell(j) = {bspline(knots(j:j+4))};
    spcell(j) = {fn2fm(ppcell{j},'B-')};
    ddppcell(j) = {fnder(ppcell{j},2)};
end

N = NaN(m,n+2);
for j=1:n+2
    N(:,j) = fnval(spcell{j},x);
end

Indx = 1;

rowIndx = []; 
columeIndx = [];
for i=1:n+2
    for j=i-3:i+3
        if j>0 && j<n+3
            pp = fnmulti(ddppcell{i},ddppcell{j});
            rowIndx(Indx) = i;
            columeIndx(Indx) = j;
            Ovalue(Indx) = diff(fnval(fnint(pp),[pp.breaks(1),pp.breaks(end)]));
            Indx = Indx + 1;
        end
    end
end
Omega = sparse(rowIndx,columeIndx,Ovalue);
end
function pp = fnmulti(f1,f2)
    break1 = f1.breaks;
    break2 = f2.breaks;
    newbreaks = intersect(break1,break2);

    n = length(newbreaks); 

    neworder = f1.order + f2.order-1;

    newcoefs = NaN(n-1,neworder);

    for i=1:n-1
        beginbreakpoint = newbreaks(i);
        Indx1 = break1(1:end-1) == beginbreakpoint;
        Indx2 = break2(1:end-1) == beginbreakpoint;
        coefs1 = f1.coefs(Indx1,:);
        coefs2 = f2.coefs(Indx2,:);
        newcoefs(i,:) = conv(coefs1,coefs2);
    end
    pp = mkpp(newbreaks,newcoefs);
end
