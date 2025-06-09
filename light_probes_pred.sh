if [ -z "$1" ]; then
  echo "Please provide a directory path. e.g. /home/yangmi/s3data-3/beauty-lvm/v2/light/768"
  exit 1
fi

# Store the directory path
directory="$1"
exposure_value='0,-0.5,-1.0,-1.5,-2.0,-2.5,-3.0,-3.5,-4.0,-4.5,-5'
output_directory="$2"
resolution=$(basename "$directory")
# Find all subfolders under the given directory
for subfolder in "$directory"/batch_*/; do
  # Check if it is a directory
  if [ -d "$subfolder" ]; then
    output_dir="$output_directory"/"$resolution"/$(basename "$subfolder")/light_mask_intensity
    if [ "$(find "$output_dir" -mindepth 1 -print -quit)" ]; then
      # the folder is not empty, skip
      echo "Skipping $subfolder"
      continue
    fi
    # Pass the subfolder name to the Python command
    echo "Processing subfolder: $subfolder"
    #rm -rf "$output_dir"
    
    #python ball2envmap.py --ball_dir "$subfolder"/square --envmap_dir "$subfolder"/envmap
    #python exposure2hdr.py --input_dir "$subfolder"/envmap --output_dir "$subfolder"/hdr --EV "$exposure_value" # optional
    #python square2chromeball.py --input_dir "$subfolder"/square --ball_dir "$subfolder"/ball
    #python exposure2hdr.py --input_dir "$subfolder"/ball --output_dir "$subfolder"/hdr_ball --EV "$exposure_value" # optional
    #python bright_map.py --input_dir "$subfolder" --output_dir "$output_directory"
    python light_probes_predict_v2.py --input_dir "$subfolder"/ball --output_dir "$output_directory"/"$resolution"/$(basename "$subfolder")/light_mask_intensity 
    #python create_light_mask.py --input_dir "$subfolder"/light_mask_intensity --output_dir "$output_dir"

    echo "Completed processing for: $subfolder"
  fi
done
# usage: bash ./light_probes_pred.sh /home/yangmi/s3data-3/beauty-lvm/v2/cropped/{img_dim} /home/yangmi/s3data-3/beauty-lvm/v2/light
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/768/batch_1//ball/afnanperfumes435246283_1121882919013095_4724912819739764924_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/768/batch_133//ball/floslekpl309364365_1092451134973180_6944744645153657895_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/768/batch_31//ball/sante_naturkosmetik318203205_542383137407398_7641134917194838590_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1024/batch_1//ball/aesturaofficial318000709_837039104278207_9062032379317012614_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1024/batch_1//ball/alphascience_singapore322382243_824538452041859_85726852497864607_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1024/batch_115//ball/dndgel11023_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1024/batch_118//ball/dsanddurga353595097_1046114306757912_1513720814865170477_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1024/batch_121//ball/egskincare11058_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1024/batch_122//ball/ellisbrooklyn10793_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1024/batch_124//ball/eneomey10018_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1024/batch_127//ball/esteelauder274265399_369297791381599_2585277197822894797_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1024/batch_130//ball/exidealexidealmini10269_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1024/batch_13//ball/cantubeauty305794091_126617003279881_5182262022399980247_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1024/batch_134//ball/focallurebeauty_us358373824_788722209380130_5545744855040908062_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1024/batch_136//ball/beyoutifulbali153775618_249989100065493_7188019556915133944_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1024/batch_136//ball/biobalicekcz28155189_199175647346350_3235982855075004416_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1024/batch_136//ball/biore_de189018322_4012151542197129_911787605516634368_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1024/batch_14//ball/caresoofficial439936461_1118270992714393_5482086563384797711_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1024/batch_2//ball/anannatokyo204222818_289650945997789_525934351547756519_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1024/batch_2//ball/apricotbeautyhealthcare291995093_598549961902532_656413383086983947_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1024/batch_25//ball/shaeri_paris69718770_381047292590605_8855043172027759219_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1024/batch_25//ball/sharperimage_com331792280_1618900221948309_5936826871994057144_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1024/batch_25//ball/sharperimage_com445714301_344982018271179_4915004350043231683_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1024/batch_29//ball/clarinsanz35934551_629162457449514_6872452383796887552_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1024/batch_30//ball/scarlett_whitening79746101_829436320836609_6296240159654134851_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1024/batch_32//ball/sam_u_jp465825038_18048613769052419_4167225302827239303_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1024/batch_4//ball/bareminerals_norge116897171_3555235407828235_2139884278816768653_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1024/batch_4//ball/beautyheroes358816167_246187484852893_2958887391296752340_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1024/batch_52//ball/perfectpotionjapan182718379_807735709869796_3435557164330726430_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1024/batch_53//ball/payot_de_at140732276_453423752502302_7385226628280484961_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1024/batch_53//ball/payotofficial74360488_410724923187457_3029321680113597963_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1024/batch_6//ball/biossancebrasil47218087_127207214966750_1780417414841344704_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1024/batch_62//ball/ogxbeautygermany270238585_316160467048074_4442276448916408074_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1024/batch_70//ball/naturisimo59796663_141629920329546_1013967149211575475_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1024/batch_71//ball/narscosmeticskorea370245992_1182846919306272_3267907200511152369_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_0//ball/accakappajapan408555523_722948376549516_8261336072064186395_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_1//ball/af94_313792440_657256862452863_4548151008782874520_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_119//ball/dysonbeauty10346_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_120//ball/ecotools10253_ev-50.pn
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_120//ball/ecotools10266_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_124//ball/emstoreskincare10364_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_126//ball/essie364390239_702920624990881_587945117011642416_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_126//ball/esteelauder_hk24845886_312248235929551_5117966330804305920_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_131//ball/farmaesthetics461845853_1536869347199712_802083307042843448_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_132//ball/fielefragrances10082_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_135//ball/forvrmood468216413_18051614197986602_2436335150146128759_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_136//ball/beterspain272686803_1006078063309754_6231660478447945266_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_136//ball/bioprogramminginternational74486581_162807038262977_6268155971745867916_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_16//ball/kerastase_official424898678_1710625562802755_6221789275014166281_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_19//ball/namcosmetics279680160_1102184127010057_8636096261747300252_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_19//ball/sosserum184269445_196256152212559_8497401063517855584_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_2//ball/arquiste364773960_18306118759102064_1344952417080581673_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_24//ball/shiro_japan130365320_316715492727124_6344588957859220305_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_26//ball/sensatiarussia115978159_828058457601954_149872537885344086_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_27//ball/sensatiarussia242795949_390212155810538_5842186417788615694_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_3//ball/banilacousa418793305_1178101493058701_3484918588924960058_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_30//ball/scarlett_whitening74880443_213977712948460_6004833215463720898_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_31//ball/saroderue240301619_356125326148145_1289975297258978006_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_32//ball/nailpolishdirect276121095_380923433592123_2473288071363444929_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_32//ball/saladcode122144697_221918189268045_4647924211614338969_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_32//ball/saladcode78712900_465641677471024_1424897126636466027_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_4//ball/bastille_parfums380857541_18018110902757682_2075757909650616044_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_49//ball/pixibeautyuk413411331_391985663213648_8999916933311795624_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_5//ball/frassai321800970_173285522065301_6265702216294502945_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_5//ball/freemanbeauty417967929_905159774458122_6287468569953300398_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_50//ball/pierrerene_professional306133097_795746624948679_3804512995891398669_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_51//ball/perriconemd297112552_1366601543836017_4339383110652621977_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_51//ball/petalfresh198718739_2943412602606227_6875951975785475582_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_55//ball/parfumsmicallef257658010_939332040018742_7122928485053912118_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_62//ball/peterthomasrothofficial260551384_1003764246844569_1513304399011307871_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_80//ball/maxaroma328733889_6254999134519765_8997575222088916482_n_ev-50.png
# No light detected /home/yangmi/s3data-3/beauty-lvm/v2/light/1440/batch_9/ball/bodyfantasies460824906_864145712450028_383995041423139527_n_ev-50.png