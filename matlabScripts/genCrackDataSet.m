%% ����mnist_uint8��һ�����ݼ�
% pathIn='D:\uProjects\AI\project4\��Ŀ4-�߽������\211028-hyperIn\';
pathRoot='E:\ML-DL\ML211028\Data\';
nker=57;
X=[];
Y=[];
for picIdx=7:14%3:22 %��ͼ���ļ���
     pathIn=[pathRoot ,'autoEdge',num2str(picIdx)];

    fileList=dir(pathIn);  %��չ��
    for i=3:length(fileList)
        if length(fileList(i).name)>6
            if strcmp(fileList(i).name(1:6),'origin')
                name=fileList(i).name(8:end-4);
            end
        end
    end
    Pic=imread([pathIn,'\origin_',name,'.bmp']);
    Im=imread([pathIn,'\xlabel_',name,'.png']); %png�������׵����ɷ�
    % for j=1:6 %��ת
    %     imgs{j}=imrotate(Im,15*j,'bilinear');
    % end
    % for j=1:6 %��ת
    %     imgs{j+6}=fliplr(imgs{j});
    % end

    markCrack=Im(:,:,1)>250 & Im(:,:,2)<250; %r
    markEdge=Im(:,:,2)>250 & Im(:,:,3)<250;  %g
    markFold=Im(:,:,3)>250 & Im(:,:,2)<250;  %b

    markAll=markCrack | markEdge | markFold;
    markPic=cat(3,markCrack,markEdge,markFold);
    
    coverPic=cat(3,Pic,Pic,Pic);
    coverPic(Im>0)=Im(Im>0);
    
    %%
    [m,n]=size(markCrack);
    haralick=0;%���޸���ѡ�������顿
    if haralick
        data=zeros(m,n,14);%haralick 14����
    else
        data=zeros(m,n,8);  %sobel 5+3����
    end
    lq=length(data(1,1,:)); %������Ŀ
    delta=floor(nker/2);
    ddx=(nker+1)*2;
    for i=ddx:2:m-ddx
        for j=ddx:2:n-ddx
            if haralick 
                if markAll(i,j)>0 %��ǹ��ĲŽ���haralickTextureFeatures����
                    I=Pic(i-delta:i+delta-1,j-delta:j+delta-1);
                    glcm = graycomatrix(I, 'offset', [0 1], 'Symmetric', true);
                    xFeatures = 1:14;
                    x = haralickTextureFeatures(glcm, xFeatures);

                    data(i,j,:)=x;
                end
            else
                if markAll(i,j)>0
                    I=Pic(i-delta:i+delta-1,j-delta:j+delta-1);
                    I=imresize(I,0.5);
                    dataNow=reshape(I,1,[]);
                    if markCrack(i,j)>0
                        yNow=[1,0,0];
                    elseif markEdge(i,j)>0
                        yNow=[0,1,0];
                    else
                        yNow=[0,0,1];
                    end
                    
                    X=[X;dataNow];
                    Y=[Y;yNow];
                end
            end
        end
    end
end

%%
X=X(1:11500,:);
Y=Y(1:11500,:);

idx=unique(floor(rand(3500,1)*11500));
idx=idx(2:3001);
idx2=1:length(X(:,1));
idx3 = setdiff(idx2,idx);

test_x=X(idx,:);
test_y=Y(idx,:);
train_x=X(idx3,:);
train_y=Y(idx3,:);
save('crack_uint8.mat','train_x','train_y','test_x','test_y');





