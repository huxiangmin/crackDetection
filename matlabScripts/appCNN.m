%% 使用已有的test_example_CNN进行分类
load crackCnn
addpath(genpath('D:/SOFTWARES/Matlab/toolbox/DeepLearnToolbox-master'))
%最好一次性判断所有边缘！！
nker=57;
% 检测效果！！  需要事先载入第一节定义的kernel。
pathRoot='E:\ML-DL\ML211028\';
pathOut=[pathRoot ,'211028-hyperOut'];%可以定义输出图像（检测结果）位置

for picIdx=27%8:14%3:22 %多图多文件夹
    X=[];
    Y=[];
    pathIn=[pathRoot ,'autoEdge',num2str(picIdx)];

    fileList=dir(pathIn);  %扩展名
    for i=3:length(fileList)
        if length(fileList(i).name)>6
            if strcmp(fileList(i).name(1:6),'origin')
                name=fileList(i).name(8:end-4);
            end
            if strcmp(fileList(i).name(1:6),'canny_')
                if ~strcmp(fileList(i).name(7),'A')
                    name1=fileList(i).name;
                end
            end
        end
    end
    Pic=imread([pathIn,'\origin_',name,'.bmp']);
    Im=imread([pathIn, '\',name1]); %读取全部边缘
    markAll=Im(:,:)>250;
    coverPic=cat(3,Pic,Pic,Pic);%初始图像，后边会修改边缘颜色
    %
    [m,n]=size(markAll);
    PIC=zeros(m,n);
    delta=floor(nker/2);
    ddx=delta+1;
    for i=ddx:m-ddx
        for j=ddx:n-ddx
            if markAll(i,j)>0
                I=Pic(i-delta:i+delta-1,j-delta:j+delta-1);
                I=imresize(I,0.5);
                X=cat(3,X,double(reshape(I,28,28))/255);
                
                PIC(i,j)=1;
            end
        end
    end
    
    net = cnnff(cnn, X);%计算、分类、标记颜色
    [~, h] = max(net.o);

    num=1;
    for i=ddx:m-ddx
        for j=ddx:n-ddx
            if PIC(i,j)==1
                if h(num)==3
                    coverPic(i,j,:)=[0,0,255]; %蓝色
                elseif h(num)==1
                    coverPic(i,j,:)=[255,0,0]; %红色
                elseif h(num)==2
                    coverPic(i,j,:)=[0,255,0]; %绿色
                end  
                num=num+1;
            end
        end
    end
    imwrite(uint8(coverPic),[pathIn,'\OutPutGoodXY_',name,'.bmp']);
end

