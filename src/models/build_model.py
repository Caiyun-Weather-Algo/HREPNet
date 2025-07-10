from src.models import condDiT,  swin_transformer_3d, swin_transformer_3d_hr
from src.models.ldm.models import autoencoder
from src.models.diffusion import create_diffusion


def build_model(model_name, cfg):
    n = len(cfg.input.levels)
    input_channels = (cfg.hyper_params.input_step) * (len(cfg.input["surface"]) + len(cfg.input["high"]) * n)
    #todo, only rainfall 
    output_channels = (cfg.hyper_params.forecast_step) * len(cfg.output["surface"])    
    
    print(f"input_channels={input_channels}, output_channels={output_channels}")
   
    if model_name == "swin_transformer_3d":
        args = cfg.model.swin_transformer_3d
        model = swin_transformer_3d.SwinTransformer3D(
                 patch_size=args.patch_size,
                 in_chans_2d=cfg.hyper_params.input_step * len(cfg.input.surface) + 9,   # TODO
                 in_chans_3d=cfg.hyper_params.input_step * len(cfg.input.high),
                 embed_dim=args.embed_dim,
                 depths=args.depths,
                 num_heads=args.num_heads,
                 window_size=args.window_size,
                 add_boundary=args.add_boundary,
                 fcst_step=cfg.hyper_params.forecast_step,
                 use_checkpoint=args.use_checkpoint, 
                 earth_position=args.earth_position,
                 mlp_embedding=args.mlp_embedding,
               #  frozen_stages=args.frozen_stages,
                 upsampler=args.upsampler
                 )
    elif model_name == "swin_transformer_3d_hr":
        args = cfg.model.swin_transformer_3d_hr
        model = swin_transformer_3d_hr.SwinTransformer3D(
                 patch_size=args.patch_size,
                 in_chans_2d=cfg.hyper_params.input_step * len(cfg.input.surface) + 9,   # TODO
                 in_chans_3d=cfg.hyper_params.input_step * len(cfg.input.high),
                 embed_dim=args.embed_dim,
                 depths=args.depths,
                 num_heads=args.num_heads,
                 window_size=args.window_size,
                 add_boundary=args.add_boundary,
                 fcst_step=cfg.hyper_params.forecast_step,
                 use_checkpoint=args.use_checkpoint, 
                 earth_position=args.earth_position,
                 mlp_embedding=args.mlp_embedding,
                 upsampler=args.upsampler
                 )   
    elif model_name == 'autoencoder_kl_gan':
        args = cfg.model.autoencoder_kl_gan
        model = autoencoder.AutoencoderKL(
            ddconfig=args['ddconfig'], 
            lossconfig=args['lossconfig'], 
            embed_dim=args['embed_dim']
            )
        
    elif model_name == 'condDiT':
        args = cfg.model.condDiT['denoiseconfig']
        model = condDiT.DiT(
            img_size=args['img_size'], 
            patch_size=args['patch_size'], 
            in_channels=args['in_channels'], 
            cond_channels=args['cond_channels'], 
            depth=args['depth'], 
            hidden_size=args['hidden_size'], 
            num_heads=args['num_heads'])
         
    elif model_name == 'diffusion': 
        # diffusion 
        args = cfg.model[cfg.model.name]['diffconfig']
        model = create_diffusion(
            timestep_respacing=args['timestep_respacing'],
            noise_schedule=args['noise_schedule'], 
            use_kl=args['use_kl'],
            sigma_small=args['sigma_small'],
            predict_xstart=args['predict_xstart'],
            learn_sigma=args['learn_sigma'],
            rescale_learned_sigmas=args['rescale_learned_sigmas'],
            diffusion_steps=args['diffusion_steps']
            )  # default: 1000 steps, linear noise schedule
    return model


