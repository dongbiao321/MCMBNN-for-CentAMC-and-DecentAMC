%生成test测试集（10分类）
clc
close all
clear all
N=10000;
Dataset(N);
function []=Dataset(N)
    L=128;
    for snr=-10:2:20
    name=['test/snr=',num2str(snr),'.mat'];
    freqsep = 2;  % Frequency separation (Hz)
    nsamp = 8;    % Number of samples per symbol
    Fs = 32;      % Sample rate (Hz)
    % k/N
    frequency_offset = zeros(128,1);
    for j = 1:128
        frequency_offset(j,1)=j/128;
    end
    IQ_2fsk=zeros(N,L,2);
    for i=1:N
        x=randi([0,1],L/8,1);
        y_com = fskmod(x,2,freqsep,nsamp,Fs);
        y_front = y_com(1:7,:);
        y_back = y_com(8:128,:);
        y_new = [y_back;y_front];
        y_com = raylrnd(1).*y_new.*exp(1i*(pi/16+pi*2*0.1*frequency_offset));
        y_noise = awgn(y_com, snr, 'measured');
        y_noise = y_noise./std(y_noise);
        x = real(y_noise);
        y = imag(y_noise);
        IQ_2fsk(i,:,1)=x;
        IQ_2fsk(i,:,2)=y;
    end  
    
    IQ_4fsk=zeros(N,L,2);
    for i=1:N
        x=randi([0,3],L/8,1);
        y_com = fskmod(x,4,freqsep,nsamp,Fs);
        y_front = y_com(1:7,:);
        y_back = y_com(8:128,:);
        y_new = [y_back;y_front];
        y_com = raylrnd(1).*y_new.*exp(1i*(pi/16+pi*2*0.1*frequency_offset)); 
        y_noise = awgn(y_com, snr, 'measured');
        y_noise = y_noise./std(y_noise);
        x = real(y_noise);
        y = imag(y_noise);
        IQ_4fsk(i,:,1)=x;
        IQ_4fsk(i,:,2)=y;
    end  
    IQ_8fsk=zeros(N,L,2);
    for i=1:N
        x=randi([0,7],L/8,1);
        y_com = fskmod(x,8,freqsep,nsamp,Fs);
        y_front = y_com(1:7,:);
        y_back = y_com(8:128,:);
        y_new = [y_back;y_front];
        y_com = raylrnd(1).*y_new.*exp(1i*(pi/16+pi*2*0.1*frequency_offset));
        y_noise = awgn(y_com, snr, 'measured');
        y_noise = y_noise./std(y_noise);
        x = real(y_noise);
        y = imag(y_noise);
        IQ_8fsk(i,:,1)=x;
        IQ_8fsk(i,:,2)=y;
    end  
    
    IQ_2psk=zeros(N,L,2);
    for i=1:N
        x=randi([0,1],L,1);
        y_com = pskmod(x,2);
        y_front = y_com(1:7,:);
        y_back = y_com(8:128,:);
        y_new = [y_back;y_front];
        y_com = raylrnd(1).*y_new.*exp(1i*(pi/16+pi*2*0.1*frequency_offset));
        y_noise = awgn(y_com, snr, 'measured');
        y_noise = y_noise./std(y_noise);
        x = real(y_noise);
        y = imag(y_noise);
        IQ_2psk(i,:,1)=x;
        IQ_2psk(i,:,2)=y;
    end
    
    IQ_4psk=zeros(N,L,2);
    for i=1:N
        x=randi([0,3],L,1);
        y_com=pskmod(x,4);
        y_front = y_com(1:7,:);
        y_back = y_com(8:128,:);
        y_new = [y_back;y_front];
        y_com = raylrnd(1).*y_new.*exp(1i*(pi/16+pi*2*0.1*frequency_offset));
        y_noise = awgn(y_com, snr, 'measured');
        y_noise = y_noise./std(y_noise);
        x = real(y_noise);
        y = imag(y_noise);
        IQ_4psk(i,:,1)=x;
        IQ_4psk(i,:,2)=y;
    end
    
    IQ_8psk=zeros(N,L,2);
    for i=1:N
        x=randi([0,7],L,1);
        y_com=pskmod(x,8);
        y_front = y_com(1:7,:);
        y_back = y_com(8:128,:);
        y_new = [y_back;y_front];
        y_com = raylrnd(1).*y_new.*exp(1i*(pi/16+pi*2*0.1*frequency_offset));
        y_noise = awgn(y_com, snr, 'measured');
        y_noise = y_noise./std(y_noise);
        x = real(y_noise);
        y = imag(y_noise);
        IQ_8psk(i,:,1)=x;
        IQ_8psk(i,:,2)=y;
    end
    
    IQ_16qam=zeros(N,L,2);
    for i=1:N
        x=randi([0,15],L,1);
        y_com=qammod(x,16);
        y_front = y_com(1:7,:);
        y_back = y_com(8:128,:);
        y_new = [y_back;y_front];
        y_com = raylrnd(1).*y_new.*exp(1i*(pi/16+pi*2*0.1*frequency_offset));
        y_noise = awgn(y_com, snr, 'measured');
        y_noise = y_noise./std(y_noise);
        x = real(y_noise);
        y = imag(y_noise);
        IQ_16qam(i,:,1)=x;
        IQ_16qam(i,:,2)=y;
    end
  
    IQ_128qam=zeros(N,L,2);
    for i=1:N
        x=randi([0,127],L,1);
        y_com=qammod(x,128);
        y_front = y_com(1:7,:);
        y_back = y_com(8:128,:);
        y_new = [y_back;y_front];
        y_com = raylrnd(1).*y_new.*exp(1i*(pi/16+pi*2*0.1*frequency_offset));
        y_noise = awgn(y_com, snr, 'measured');
        y_noise = y_noise./std(y_noise);
        x = real(y_noise);
        y = imag(y_noise);
        IQ_128qam(i,:,1)=x;
        IQ_128qam(i,:,2)=y;
    end
    
    IQ_256qam=zeros(N,L,2);
    for i=1:N
        x=randi([0,255],L,1);
        y_com=qammod(x,256);
        y_front = y_com(1:7,:);
        y_back = y_com(8:128,:);
        y_new = [y_back;y_front];
        y_com = raylrnd(1).*y_new.*exp(1i*(pi/16+pi*2*0.1*frequency_offset));
        y_noise = awgn(y_com, snr, 'measured');
        y_noise = y_noise./std(y_noise);
        x = real(y_noise);
        y = imag(y_noise);
        IQ_256qam(i,:,1)=x;
        IQ_256qam(i,:,2)=y;
    end
    
    IQ=cat(1,IQ_2fsk,IQ_4fsk,IQ_8fsk,IQ_2psk,IQ_4psk,IQ_8psk,IQ_16qam,IQ_128qam,IQ_256qam);
    save(name,'IQ')  
    end
end