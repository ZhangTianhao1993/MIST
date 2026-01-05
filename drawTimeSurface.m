function drawTimeSurface(ROIZvalue,CData,fv,mymap,minZ,maxZ,figurename)
    n = size(mymap,1);
    hn = round(n/2);
    y = linspace(minZ,maxZ,n);
    nmymap = NaN(n,3);
    for i=1:n
        if y(i)>=0
            
            nmymap(i,:) = mymap(round(hn+hn*y(i)/maxZ),:);
        else
            nmymap(i,:) = mymap(round(hn*(y(i)-minZ)/abs(minZ))+1,:);
        end
    end
    mymap = nmymap;
    
     faceNum = length(CData);
     FaceColor = NaN(faceNum,1);
     uniqueCData = unique(CData);
     n = length(uniqueCData);
     for i=1:n
        val = uniqueCData(i);
        I = cell2mat(ROIZvalue(:,2)) == val;    
        zvalue = ROIZvalue{I,3};
        if ~isinf(zvalue)
            FaceColor(CData == uniqueCData(i)) = zvalue;
        end
    end
    f = figure('Color','White','Units','centimeters','Position',[10,10,5.5,3.5]);
    %factor = 0.93;
    factor = 1.1;
    colormap(f,mymap);
    
    
    ax1 = subplot('Position',[0,0.5,0.5,0.5]);
    createPatch(fv,ax1,FaceColor);
    campos(ax1,[100,-20,20]);
    camtarget(ax1,[0,-20,20]);
    camzoom(ax1,factor);
    light('Position',[1,0,0]);
    daspect([1,1,1]);
    axis off
    caxis([minZ maxZ]);
    
    ax2 = subplot('Position',[0.5,0.5,0.5,0.5]);
    createPatch(fv,ax2,FaceColor);
    campos(ax2,[-100,-20,20]);
    camtarget(ax2,[0,-20,20]);
    camzoom(ax2,factor);
    light('Position',[-1,0,0]);
    daspect([1,1,1]);
    axis off
    caxis([minZ maxZ]);
    
    ax3 = subplot('Position',[0,0,0.5,0.5]);
    createPatch(fv,ax3,FaceColor);
    campos(ax3,[0,-20,20]);
    camtarget(ax3,[100,-20,20]);
    camzoom(ax3,factor);
    light('Position',[-1,0,-0.5]);
    daspect([1,1,1]);
    axis off
    caxis([minZ maxZ]);
    
    ax4 = subplot('Position',[0.5,0,0.5,0.5]);
    createPatch(fv,ax4,FaceColor);
    campos(ax4,[0,-20,20]);
    camtarget(ax4,[-100,-20,20]);
    camzoom(ax4,factor);
    light('Position',[1,0,-0.5]);
    daspect([1,1,1]);
    axis off
    caxis([minZ maxZ]);
    
   exportgraphics(f,figurename,'ContentType','image','Resolution',1000);
end
function createPatch(fv,ax,FaceColor)
    pback = patch(ax,fv);
    pback.FaceVertexCData = [0.9 0.9 0.9];
    pback.EdgeAlpha = 0;
    pback.FaceColor = 'flat';
    material(pback,'dull');
    %pback.CDataMapping = 'direct';
    p1 = patch(ax,fv);
    p1.FaceVertexCData = FaceColor;
    p1.EdgeAlpha = 0;
    p1.FaceColor = 'flat';
    material(p1,'dull');
end