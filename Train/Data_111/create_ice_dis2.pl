

#genice2 ice1h --rep 2 2 2 --shift 1.322 0.123 3.376 --add_noise 1.0212 --seed 2123 > ice1h.gro

my $num_gro = 100;
for (my $i = 0; $i < $num_gro; $i++) {
	my $rand_shift1 = 6*(rand());
	my $rand_shift2 = 6*(rand());
	my $rand_shift3 = 6*(rand());
	#
	my $rand_noise = 1.61;
	my $rand_seed = int(5106*(rand()));
	#
	system("cd displacement && genice2 ice1h --rep 5 5 5 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 6.555.$rand_seed.ice1h.gro");
	#system("cd displacement && genice2 ice1h --rep 4 3 4 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 6.434.$rand_seed.ice1h.gro");
	#system("cd displacement && genice2 ice1h --rep 4 3 2 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 6.432.$rand_seed.ice1h.gro");
	#	save_path	     type     cry_rep									   type.cry_rep.i	type
	system("cd displacement && genice2 ice1c --rep 5 5 5 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 5.555.$rand_seed.ice1c.gro");
	#system("cd displacement && genice2 ice1c --rep 4 3 4 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 5.434.$rand_seed.ice1c.gro");
	#system("cd displacement && genice2 ice1c --rep 4 3 2 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 5.432.$rand_seed.ice1c.gro");
	#       save_path            type     cry_rep                                                                      type.cry_rep.i       type
	system("cd displacement && genice2 ice2 --rep 5 5 5 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 7.555.$rand_seed.ice2.gro");
	#system("cd displacement && genice2 ice2 --rep 4 3 4 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 7.434.$rand_seed.ice2.gro");
	#system("cd displacement && genice2 ice2 --rep 4 3 2 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 7.432.$rand_seed.ice2.gro");
        #       save_path            type     cry_rep                                                                      type.cry_rep.i       type
	system("cd displacement && genice2 ice6 --rep 5 5 5 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 13.555.$rand_seed.ice6.gro");
	#system("cd displacement && genice2 ice6 --rep 4 3 4 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 13.434.$rand_seed.ice6.gro");
	#system("cd displacement && genice2 ice6 --rep 4 3 2 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 13.432.$rand_seed.ice6.gro");
        #       save_path            type     cry_rep                                                                      type.cry_rep.i       type
	#	system("cd displacement && genice2 CS1 --rep 4 4 4 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 1.333.$rand_seed.ECS1.gro");
	#        system("cd displacement && genice2 CS1 --rep 4 3 4 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 1.411.$rand_seed.ECS1.gro");
	#        system("cd displacement && genice2 CS1 --rep 4 3 2 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 1.431.$rand_seed.ECS1.gro");
        #       save_path            type     cry_rep                                                                      type.cry_rep.i       type
#	system("cd displacement && genice2 CS1 -g 12=ch4 -g 14=ch4 --rep 3 3 3 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 1.333.$rand_seed.CS1.gro");
	#        system("cd displacement && genice2 CS1 -g 12=ch4 -g 14=ch4 --rep 4 1 1 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 1.411.$rand_seed.CS1.gro");
	#        system("cd displacement && genice2 CS1 -g 12=ch4 -g 14=ch4 --rep 4 3 1 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 1.431.$rand_seed.CS1.gro");
        #       save_path            type     cry_rep                                                                      type.cry_rep.i       type
	#	system("cd displacement && genice2 CS2 --rep 3 3 3 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 2.333.$rand_seed.ECS2.gro");
	#        system("cd displacement && genice2 CS2 --rep 4 1 1 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 2.411.$rand_seed.ECS2.gro");
	#        system("cd displacement && genice2 CS2 --rep 4 3 1 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 2.431.$rand_seed.ECS2.gro");
        #       save_path            type     cry_rep                                                                      type.cry_rep.i       type
	#	system("cd displacement && genice2 CS2 -g 12=ch4 -g 16=ch4 --rep 3 3 3 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 2.333.$rand_seed.CS2.gro");
	#        system("cd displacement && genice2 CS2 -g 12=ch4 -g 16=ch4 --rep 4 1 1 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 2.411.$rand_seed.CS2.gro");
	#        system("cd displacement && genice2 CS2 -g 12=ch4 -g 16=ch4 --rep 4 3 1 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 2.431.$rand_seed.CS2.gro");
        #       save_path            type     cry_rep                                                                      type.cry_rep.i       type
	#	system("cd displacement && genice2 ST --rep 3 3 3 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 4.333.$rand_seed.ST.gro");
	#        system("cd displacement && genice2 ST --rep 4 1 1 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 4.411.$rand_seed.ST.gro");
	#        system("cd displacement && genice2 ST --rep 4 3 1 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 4.431.$rand_seed.ST.gro");
        #       save_path            type     cry_rep                                                                      type.cry_rep.i       type
	system("cd displacement && genice2 ice3 --rep 5 5 5 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 9.555.$rand_seed.ice3.gro");
	#system("cd displacement && genice2 ice3 --rep 4 3 4 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 9.434.$rand_seed.ice3.gro");
	#system("cd displacement && genice2 ice3 --rep 4 3 2 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 9.432.$rand_seed.ice3.gro");
        #       save_path            type     cry_rep                                                                      type.cry_rep.i       type
	system("cd displacement && genice2 ice4 --rep 5 5 5 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 10.555.$rand_seed.ice4.gro");
	#system("cd displacement && genice2 ice4 --rep 4 3 4 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 10.434.$rand_seed.ice4.gro");
	#system("cd displacement && genice2 ice4 --rep 4 3 2 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 10.432.$rand_seed.ice4.gro");
        #       save_path            type     cry_rep                                                                      type.cry_rep.i       type
	system("cd displacement && genice2 ice5 --rep 5 5 5 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 12.555.$rand_seed.ice5.gro");
	#system("cd displacement && genice2 ice5 --rep 4 3 4 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 12.434.$rand_seed.ice5.gro");
	#system("cd displacement && genice2 ice5 --rep 4 3 2 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 12.432.$rand_seed.ice5.gro");
        #       save_path            type     cry_rep                                                                      type.cry_rep.i       type
	system("cd displacement && genice2 ice7 --rep 5 5 5 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 14.555.$rand_seed.ice7.gro");
	#system("cd displacement && genice2 ice7 --rep 4 3 4 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 14.434.$rand_seed.ice7.gro");
	#system("cd displacement && genice2 ice7 --rep 4 3 2 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 14.432.$rand_seed.ice7.gro");
        #       save_path            type     cry_rep                                                                      type.cry_rep.i       type
	system("cd displacement && genice2 ice8 --rep 5 5 5 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 15.555.$rand_seed.ice8.gro");
	#system("cd displacement && genice2 ice8 --rep 4 3 4 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 15.434.$rand_seed.ice8.gro");
	#system("cd displacement && genice2 ice8 --rep 4 3 2 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 15.432.$rand_seed.ice8.gro");
        #       save_path            type     cry_rep                                                                      type.cry_rep.i       type
	system("cd displacement && genice2 ice9 --rep 5 5 5 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 16.555.$rand_seed.ice9.gro");
	#system("cd displacement && genice2 ice9 --rep 4 3 4 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 16.434.$rand_seed.ice9.gro");
	#system("cd displacement && genice2 ice9 --rep 4 3 2 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 16.432.$rand_seed.ice9.gro");
        #       save_path            type     cry_rep                                                                      type.cry_rep.i       type
	system("cd displacement && genice2 ice0 --rep 5 5 5 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 17.555.$rand_seed.ice0.gro");
	#system("cd displacement && genice2 ice0 --rep 4 3 4 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 17.434.$rand_seed.ice0.gro");
	#system("cd displacement && genice2 ice0 --rep 4 3 2 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 17.432.$rand_seed.ice0.gro");
        #       save_path            type     cry_rep                                                                      type.cry_rep.i       type
	system("cd displacement && genice2 ice11 --rep 5 5 5 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 18.555.$rand_seed.ice11.gro");
	#system("cd displacement && genice2 ice11 --rep 4 3 4 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 18.434.$rand_seed.ice11.gro");
	#system("cd displacement && genice2 ice11 --rep 4 3 2 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 18.432.$rand_seed.ice11.gro");
        #       save_path            type     cry_rep                                                                      type.cry_rep.i       type
	system("cd displacement && genice2 ice12 --rep 5 5 5 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 20.555.$rand_seed.ice12.gro");
	#system("cd displacement && genice2 ice12 --rep 4 3 4 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 20.434.$rand_seed.ice12.gro");
	#system("cd displacement && genice2 ice12 --rep 4 3 2 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 20.432.$rand_seed.ice12.gro");
        #       save_path            type     cry_rep                                                                      type.cry_rep.i       type
	system("cd displacement && genice2 ice13 --rep 5 5 5 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 21.555.$rand_seed.ice13.gro");
	#system("cd displacement && genice2 ice13 --rep 4 3 4 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 21.434.$rand_seed.ice13.gro");
	#system("cd displacement && genice2 ice13 --rep 4 3 2 --shift $rand_shift1 $rand_shift2 $rand_shift3 --seed $rand_seed > 21.432.$rand_seed.ice13.gro");
        #       save_path            type     cry_rep                                                                      type.cry_rep.i       type
	#system("cd ice1c-8 && genice2 ice1c --rep 1 3 3 --shift $rand_shift1 $rand_shift2 $rand_shift3 --add_noise $rand_noise --seed $rand_seed > 5.133.$rand_seed.noise_1.7.ice1c.gro");
	#system("cd ice1c-8 && genice2 ice1c --rep 1 1 4 --shift $rand_shift1 $rand_shift2 $rand_shift3 --add_noise $rand_noise --seed $rand_seed > 5.114.$rand_seed.noise_1.7.ice1c.gro");
}
