"""
Created by Ahmad Obeid and Hasan AlMarzouqi 
for the implementation of paper:
    A Method for Explaining CRISPR/Cas12a Predictions, 
    and an Efficiency-aware gRNA Sequence Generator
    
"""

from utils import *
import warnings

class CustomVariationalLayer(keras.layers.Layer):    
    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        kl_loss = -5e-4 * K.mean( 
        1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        # beta = inputs[2]
        loss = self.vae_loss(x, z_decoded) #, beta
        self.add_loss(loss, inputs=inputs)
        return x

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    

    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--trained_mdl', default='1', type=str)
    parser.add_argument('--ready_synth', default='1', type=str)
    parser.add_argument('--heatmaps', default='0', type=str) 
    parser.add_argument('--MSM', default='1', type=str) 
    parser.add_argument('--CAM', default='1', type=str) 
    
    
    args = parser.parse_args()
    check_inputs(args.__dict__.values())
    """# Data Loading"""
    #Get HT1-1, HT1-2, HT1, HT3
    SEQ11, SEQ12, SEQ2, SEQ3, Rates11, Rates12, Rates2, Rates3 = load_data()
    num_classes = len(np.unique(Rates11))
    #Combine HT1-1 and HT1-2 to get HT1
    SEQ1, Rates1 = np.concatenate((SEQ11,SEQ12),axis=0), np.concatenate((Rates11,Rates12),axis=0)
    lb = preprocessing.LabelBinarizer().fit(Rates1)
    Rates1, Rates2, Rates3 = lb.transform(Rates1), lb.transform(Rates2), lb.transform(Rates3)
    
    #Removing TTT in PAM, important as CRISPR-VAE was trained as such
    SEQ1_nopam = np.concatenate((SEQ1[:,:4,:,:],SEQ1[:,7:,:,:]),axis=1)

    print("Data loaded successfully.\n\n")
    """# CRISPR-VAE Initialization"""
    
    seq_shape =  (31,4,1) 
    batch_size = 512
    latent_dim = 2
    
    
    input_seq = keras.Input(shape=seq_shape)
    x = layers.Conv2D(80, (5,1),activation='relu')(input_seq)
    shape_before_flattening = K.int_shape(x)
    x = layers.Flatten()(x)
    
    
    dense1 = layers.Dense(80,activation='relu')
    x = dense1(x)
    
    condition_shape = (num_classes) 
    input_condition = keras.Input(shape=condition_shape)
    x = tf.keras.layers.concatenate([x, input_condition], axis=-1)
    
    dense2 = layers.Dense(40,activation='relu')
    x = dense2(x)
    
    dense2_2 = layers.Dense(40,activation='relu')
    x = dense2_2(x)
    
    dense3 = layers.Dense(latent_dim)
    z_mean = dense3(x)
    z_log_var = layers.Dense(latent_dim)(x)
    
    
    z = layers.Lambda(sampling)([z_mean, z_log_var])
    decoder_input = layers.Input((latent_dim+num_classes))
    
    x = layers.Dense(40,activation='relu')(decoder_input)
    x = layers.Dense(40,activation='relu')(x)
    x = layers.Dense(80,activation='relu')(x)
    x = layers.Dense(np.prod(shape_before_flattening[1:]),activation='relu')(x)
    x = layers.Reshape(shape_before_flattening[1:])(x)
    x = layers.Conv2DTranspose(1, (5,1),activation='sigmoid')(x)
    
    
    decoder = Model(decoder_input, x)
    z = tf.keras.layers.concatenate([z, input_condition], axis=-1)
    z_decoded = decoder(z)
    
    y = CustomVariationalLayer()([input_seq, z_decoded])
    vae = Model([input_seq,input_condition], y)
    
    vae.compile(optimizer='adam',loss=None)
    
    if bool(int(args.trained_mdl)):
        vae.load_weights('./Files/cvae_latest.h5')
        decoder.load_weights('./Files/cvae_dec_latest.h5')
        
        print("\nCRISPR-VAE built. Weights loaded.\n\n")
    else: 
        print("Augmenting the training data...\n\n")
        SEQ_aug1, Rates_aug1 = data_aug_pro(SEQ1,Rates1.flatten())
        Rates_aug1 = lb.transform(Rates_aug1[:,None])
        SEQ_aug1_nopam = np.concatenate((SEQ_aug1[:,:4,:,:],SEQ_aug1[:,7:,:,:]),axis=1)
        print("Training CRISPR-VAE...\n\n")
        history = vae.fit(x=[SEQ_aug1_nopam,Rates_aug1],
                    y=none,batch_size=batch_size,shuffle=True,epochs=100,verbose=2) 
    
        print("End of training.\n\n")
        
    """# seq-DeepCpf1 Initialization"""
    
    seqDeepCpf = bld_seqDeepCpf()
    seqDeepCpf.load_weights('./Files/Rgs.h5')
    print("seqDeepCpf built. Weights loaded\n\n")
    
    
    """# Decoder Visualization"""
    
    from tensorflow.keras import models
    latent_output =  vae.layers[-3].output 
    latent_mdl = models.Model(inputs=vae.input, outputs=latent_output)
    latent_code = latent_mdl.predict([SEQ1_nopam,Rates1])
    
    decoded_output = decoder.layers[-1].output
    decoder_mdl = models.Model(inputs=decoder.input, outputs=decoded_output)
    decoded_seq = decoder_mdl.predict(latent_code)
    
    decoded_seq = inject_T(decoded_seq) #Get TTT in PAM  
    _, decoded_idx = one_hot(decoded_seq[879, :, :, 0]) # / #879 chosen at random, pick any other number at will
    fig1 = seq_to_img(decoded_idx) 
    
    _, input_idx = one_hot(SEQ1[879, :, :, 0]) #879 chosen at random, pick any other number at will
    fig2 = seq_to_img(input_idx)
    
    figure = np.concatenate((fig1,fig2),axis=0)    
    plt.figure(figsize=(10, 10))
    plt.axis('off')  
    plt.title('Reconstruction (up), Original (down)')  
    plt.imshow(figure)
    plt.savefig('./Files/outputs/Reconrstuction_example.png')
    plt.close('all')
    print("Reconstruction visulaization done. Check ./Files/outputs/Reconstruction_example.png\n\n")
    
    
    """# Optional: Generating Synthetic Data + Labeling by seqDeepCpf1
    
    This is optional, as the synthetic data generated for the paper is provided
    """
    if not bool(int(args.ready_synth)):
        lb2 = preprocessing.LabelBinarizer().fit(np.arange(4))
        grid_size = 100
        dims = 2
        clss = [0,99] 
        
        for cls in clss:
          decoder_mdl = models.Model(inputs=decoder.input, outputs=decoded_output)
          extent = norm.ppf(np.linspace(0.05, 0.95,grid_size),scale=1)
        
          n = grid_size**dims 
          grid = np.meshgrid(extent,extent) #for 2D
        
          flat_grid = np.reshape(grid,(dims,grid_size**dims))
          flat_grid = flat_grid.transpose()
        
          conditions = np.repeat(lb.transform(np.array([cls,])),n,axis=0)
          flat_grid = np.concatenate((flat_grid,conditions),axis=1) 
          decoded_seq = decoder_mdl.predict(flat_grid)  
        
          decoded_seq = inject_T(decoded_seq) 
          txt_array = np.array(seq_to_txt_bulk(decoded_seq.squeeze())) 
          tmp = decoded_seq.squeeze().argmax(-1).flatten()
          tmp = lb2.transform(tmp)  
          decoded_seq = np.reshape(tmp,(n,34,4))
          Rates_synth = seqDeepCpf.predict(decoded_seq)
          
          sio.savemat('./Files/synthetic_'+str(cls)+'_efficiency.mat',{'decoded_seq':decoded_seq}) #one-hot encoded
          sio.savemat('./Files/synthetic_'+str(cls)+'_efficiency_txt.mat',{'txt_array':txt_array}) #strings
          sio.savemat('./Files/synthetic_'+str(cls)+'_efficiency_labels.mat',{'Rates_synth':Rates_synth}) #Labels
          print("New synthetic data generated.\n\n")
    else: print("Using existing synthetic data.\n\n")
      
    """# Agreement Testing"""
    print("Testing the agreewment beteen CRISPR-VAE and sepqDeepCf1..\n\n")
    clss = [0,99] 
    
    for cls in clss:
      Rates = sio.loadmat('./Files/synthetic_'+str(cls)+'_pred.mat')['Rates_synth']
      plt.hist(Rates,bins=100)
    plt.title('seqDeepCpf prediction of the synthetic data')
    plt.xlabel('Prediction')
    plt.ylabel('Count')
    plt.savefig('./Files/outputs/agreement.png')
    get_agreement(0,99,None)
    print("Agreements plots generated. Check ./Files/outputs/agreement.png\n\n")
    
    
    """# Structure Confirmation"""
    if bool(int(args.heatmaps)):
        print("Getting structure heatmaps figures...\n")
        n = 100
        decoded_seq = sio.loadmat('./Files/synthetic_99_pred.mat')['decoded_seq']
        seqs_num_encode = decoded_seq.argmax(-1)
        seqs_num_encode = np.reshape(seqs_num_encode,(n,n,34))
        
        m = [3,11,29]
        mid_last = int((m[-1]-1)/2)
        out = np.zeros((n-2*mid_last,n-2*mid_last,3))
        for idx, i in enumerate(m):
          mid = int((i-1)/2)
          temp = conv(seqs_num_encode,i)
          if mid != mid_last:
            out[:,:,idx] = temp[mid_last-mid:-(mid_last-mid),mid_last-mid:-(mid_last-mid)]
          else:
            out[:,:,idx] = temp
        
        extent = norm.ppf(np.linspace(0.05, 0.95,100),scale=2.3)[mid:-mid]
        grid = np.meshgrid(extent,extent)
        for lam in m:
            plt.scatter(grid[0],grid[1],c=out[:,:,lam]) 
            plt.colorbar()
            plt.savefig('./Files/output/heatmap_'+str(lam)+'.png')
        print("Structure heatmaps figures generated. Check ./Files/outputs/heatmaps_delta.png\n\n")
        
        
    """# Method 1 for Feature Extraction"""
    
    if bool(int(args.MSM)):
        wanted_region = ['pre_pam','pam', 'seed1','seed2','seed3','seed4',
                      'tr1','tr2','tr3','tr4','tr5','tr6','tr7','tr8','tr9','tr10',
                      'prom1','prom2','prom3','post_seq']
        
        """1.   Histogram-summarize HT1, HT2, HT3"""
        
        files = ['HT1','HT2','HT3']
        choice = ['high','low']
        print("Initializing the HT data MERs...\n\n")
        for fil in files:
            for idx,_ in enumerate(choice):
              seqs_str = sio.loadmat('./Files/'+fil+'_text.mat')['txt_array']
              freq = sio.loadmat('./Files/'+fil+'_text.mat')['freq']
              seqs_wanted = seqs_str[freq>75] if choice[idx] == 'high' else seqs_str[freq<25]
              mer_size = 3
              for region in wanted_region:
                  hist_mers(seqs_wanted,region,k=mer_size,save=1,location=None,name='HT_'+choice[idx]) #
    
        """
        2.   Histogram-summarize the synthetic data
        """
        print("Initializing the synthetic data MERs...\n\n")
        n = 100 #number of points in each axis of the grid
        extent = norm.ppf(np.linspace(0.05, 0.95,n),scale=1)
        for cls in clss:
          seqs_str = sio.loadmat('./Files/synthetic_'+str(cls)+'_txt.mat')['txt_array']
          #For non-quadrant-based:
          location = (extent[0],extent[-1],extent[0],extent[-1])     
          region, wanted_location = get_region_2d(location,extent)   
          wanted_seqs = seqs_str[wanted_location]
          mer_size = 3
          for region in wanted_region:
                      hist_mers(wanted_seqs,region,k=mer_size,save=1,location=location,name='synth'+str(cls))                
              
          #For quadrant-based   
          ends = [extent[0],0]
          for x in ends:
              x1 = x + extent[-1]
              for y in ends:            
                  y1 = y + extent[-1]
                  location = (x,x1,y,y1)  
                  region, wanted_location = get_region_2d(location,extent) 
                  wanted_seqs = seqs_str[wanted_location]    
                  for region in wanted_region:
                      hist_mers(wanted_seqs,region,k=mer_size,save=1,location=location,name='synth'+str(cls))
        
        """3.   Filtering + MSMs
        
        
        """
        print("Filtering MERs...\n\n")
        missed_out(3) #colors what exists in HT2, HT3, and not HT1 in blue
        specials = highlight(3) #colors what's in blue in pink, if exists in synthetic data
        specials = collaps_special(specials)
        dic = filter_feats(3).in_high() #colors what's above the threshold by red
        draw_significant_feats(dic,circular=True,specials=specials) #Gets the MSMs
        print("MSMs are generated. Check ./Files/outputs/MSM_Qn\n\n")
        # Note: if filter_feats()  back empty, this naturally throws an error when drawing the MSMs. 
        # This usually happens with the non-quadrant-based MSM. Nevertheless, the other MSMs
        # will have been drawn
    
        """# Method 2 for Feature Extraction
        
        1.   Get the synthetic data
        """
    if bool(int(args.CAM)):
        print("Starting CAM generation...\n\n")
        SEQ_1 = sio.loadmat('./Files/synthetic_99_.mat')['decoded_seq']  
        SEQ_1_str = sio.loadmat('./Files/synthetic_99_txt.mat')['txt_array'] 
        _, a = np.unique(SEQ_1_str,return_index=True)
        SEQ_1_unique, SEQ_1_str_unique = SEQ_1[a], SEQ_1_str[a]
        
        SEQ_0 = sio.loadmat('./Files/synthetic_0_.mat')['decoded_seq']  
        
        SEQ_0_str = sio.loadmat('./Files/synthetic_0_txt.mat')['txt_array']  
        _, b = np.unique(SEQ_0_str,return_index=True)
        SEQ_0_unique, SEQ_0_str_unique = SEQ_0[b], SEQ_0_str[b]
        
        x, x_str = np.concatenate((SEQ_0_unique,SEQ_1_unique),axis=0), np.concatenate((SEQ_0_str_unique,SEQ_1_str_unique),axis=0)
        y = np.concatenate((np.zeros(len(SEQ_0_unique),),np.ones(len(SEQ_1_unique),)),axis=0)
        x, x_str, y = shuffle(x,x_str,y,random_state=123)
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33)
        
        x_train_noPAM = np.concatenate((x_train[:,:4,:],x_train[:,7:,:]),axis=1)
        x_test_noPAM = np.concatenate((x_test[:,:4,:],x_test[:,7:,:]),axis=1)
        
        """
        2.   Train binary classifier"""
        
        classifier = build_classifier_cam()
        classifier.compile('adam',loss='binary_crossentropy',metrics=['acc'])
        classifier.fit(x_train_noPAM[...,None],y_train,
                        epochs=10,verbose=2,
                        validation_data=(x_test_noPAM[...,None],y_test))
        
        """
        3.   Prepare CAM model
        
        """
        
        output = classifier.output[:,0]
        last_conv_layer = classifier.get_layer('conv2')
        grads = K.gradients(output,last_conv_layer.output)[0]
        pooled_grads = K.mean(grads,axis=(0,1,2))
        iterate = K.function([classifier.input],[pooled_grads, last_conv_layer.output[0]])
        
        """4.   Get the CAMs"""
        
        import cv2
        import matplotlib
        Quads = ['Q3','Q2','Q4','Q1']
        n = 100
        extent = norm.ppf(np.linspace(0.05, 0.95,n),scale=1)
        ends = [extent[0],0]
        cnt = 0
        for x in ends:
            x1 = x + extent[-1]
            for y in ends:            
                y1 = y + extent[-1]
                location = (x,x1,y,y1)  
                region, wanted_location = get_region_2d(location,extent) 
                wanted_seqs = SEQ_1[wanted_location]
                wanted_seqs = np.concatenate((wanted_seqs[:,:4,:],wanted_seqs[:,7:,:]),axis=1) #for without the PAM
        
                all = []
                for idx, seq in enumerate(wanted_seqs):  
                  pooled_grads_value, conv_layer_output_value = iterate([seq[None,:,:,None]]) 
                  conv_layer_output_value *= pooled_grads_value
                  heatmap = np.mean(conv_layer_output_value, axis=-1)
                  heatmap = np.maximum(heatmap, 0)
                  heatmap /= np.max(heatmap)
                  heatmap= cv2.resize(heatmap, (4, 31)) 
                  superimposed_img = heatmap * 0.4 + seq
                  if len(np.argwhere(np.isnan(heatmap))) > 0: continue
                  all += [heatmap]
                all = np.array(all)        
                avg = np.mean(all,axis=0)        
                        
                plt.matshow(avg.transpose(),cmap='Reds')
                plt.vlines([3.5,4.5,10.5,22.5,27.5],[-0.5,-0.5,-0.5,-0.5,-0.5],[4.5,4.5,4.5,4.5,4.5])
                plt.colorbar() 
                plt.savefig('./Files/outputs/CAM_'+Quads[cnt]+'.png')
                cnt += 1
        print("CAMs are generated. Check ./Files/outputs/CAMs_Qn.png")
