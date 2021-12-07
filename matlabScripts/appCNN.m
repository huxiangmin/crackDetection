%% ʹ�����е�test_example_CNN���з���
load crackCnn
addpath(genpath('D:/SOFTWARES/Matlab/toolbox/DeepLearnToolbox-master'))
%���һ�����ж����б�Ե����
nker=57;
% ���Ч������  ��Ҫ���������һ�ڶ����kernel��
pathRoot='E:\ML-DL\ML211028\';
pathOut=[pathRoot ,'211028-hyperOut'];%���Զ������ͼ�񣨼������λ��

for picIdx=27%8:14%3:22 %��ͼ���ļ���
    X=[];
    Y=[];
    pathIn=[pathRoot ,'autoEdge',num2str(picIdx)];

    fileList=dir(pathIn);  %��չ��
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
    Im=imread([pathIn, '\',name1]); %��ȡȫ����Ե
    markAll=Im(:,:)>250;
    coverPic=cat(3,Pic,Pic,Pic);%��ʼͼ�񣬺�߻��޸ı�Ե��ɫ
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
    
    net = cnnff(cnn, X);%���㡢���ࡢ�����ɫ
    [~, h] = max(net.o);

    num=1;
    for i=ddx:m-ddx
        for j=ddx:n-ddx
            if PIC(i,j)==1
                if h(num)==3
                    coverPic(i,j,:)=[0,0,255]; %��ɫ
                elseif h(num)==1
                    coverPic(i,j,:)=[255,0,0]; %��ɫ
                elseif h(num)==2
                    coverPic(i,j,:)=[0,255,0]; %��ɫ
                end  
                num=num+1;
            end
        end
    end
    imwrite(uint8(coverPic),[pathIn,'\OutPutGoodXY_',name,'.bmp']);
end

