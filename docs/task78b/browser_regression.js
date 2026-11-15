async page => {
  const credentials = __TASK78B_CREDENTIALS__;
  const base = 'http://127.0.0.1:18072';
  const out = '/home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/docs/task78b/screenshots/';
  const checks = [], layouts = [], shots = [], errors = [], resources = [];
  const widths = [{width:1440,height:900},{width:1366,height:768},{width:390,height:844},{width:375,height:667}];
  const check = (name,ok,detail) => { checks.push({name,pass:!!ok,detail}); if (!ok) throw Error(name); };
  page.on('pageerror', error => errors.push(String(error)));
  page.on('response', response => { if (response.url().startsWith(base+'/static/')) resources.push({url:response.url(),status:response.status()}); });
  async function shot(name, fullPage=false) {
    await page.screenshot({path:out+name,fullPage});
    shots.push({file:name,url:page.url(),viewport:page.viewportSize()});
  }
  async function layout(name) {
    const value=await page.evaluate(() => {
      const r = element => {if(!element)return null;const b=element.getBoundingClientRect();return {left:b.left,top:b.top,right:b.right,bottom:b.bottom,width:b.width,height:b.height};};
      return {width:innerWidth,height:innerHeight,scrollWidth:document.documentElement.scrollWidth,card:r(document.querySelector('.login-panel')),header:r(document.querySelector('header')),brand:r(document.querySelector('.brand')),menu:r(document.querySelector('nav')),account:r(document.querySelector('.account')),input:r(document.querySelector('#id_username')),button:r(document.querySelector('.login-panel button[type=submit]')),switch:r(document.querySelector('.auth-switch'))};
    });
    layouts.push({name,url:page.url(),...value});
    check(name+'_no_horizontal_overflow',value.scrollWidth<=value.width,value);
    return value;
  }
  async function loadedImages(name) {
    await page.locator('img').evaluateAll(images => Promise.all(images.filter(i=>i.loading!=='lazy'||i.getBoundingClientRect().top<innerHeight).map(i=>i.complete?Promise.resolve():new Promise(resolve=>{i.addEventListener('load',resolve,{once:true});i.addEventListener('error',resolve,{once:true});setTimeout(resolve,10000);})))) ;
    const data=await page.locator('img').evaluateAll(images=>images.map(i=>({url:i.currentSrc,loaded:i.complete&&i.naturalWidth>0,visible:i.getBoundingClientRect().top<innerHeight})));
    check(name+'_visible_real_posters_loaded',data.filter(i=>i.visible).every(i=>i.loaded)&&data.some(i=>i.loaded),data);
  }
  try {
    for (const v of widths) {
      await page.setViewportSize(v);await page.goto(base+'/cinema/login/',{waitUntil:'networkidle'});
      const k='login_'+v.width+'x'+v.height;
      const l=await layout(k);
      check(k+'_compact_auth_header',await page.getByRole('navigation').count()===0 && l.header.height<=72,l.header);
      check(k+'_no_duplicate_login_link',await page.getByRole('link',{name:'登录',exact:true}).count()===0);
      check(k+'_usable_input_height',l.input.height>=44,l.input);
      if(v.width>1000) check(k+'_whole_card_and_register_visible',l.card.top>=l.header.bottom && l.card.bottom<=v.height && l.switch.bottom<=v.height,l.card);
      await page.getByLabel('账号',{exact:true}).focus();check(k+'_input_focus',await page.locator('#id_username').evaluate(e=>e===document.activeElement));
      if(v.width===1440)await shot('01_login_desktop.png');
      if(v.width===390)await shot('02_login_mobile.png');
      if(v.width===1366)await shot('06_login_1366.png');
      if(v.width===375)await shot('07_login_375.png',true);
    }
    await page.getByLabel('账号',{exact:true}).fill('task78b_invalid');await page.getByLabel('密码',{exact:true}).fill('invalid-login-only');await page.getByRole('button',{name:'登录',exact:true}).click();
    check('invalid_login_error_readable',await page.getByRole('alert').isVisible());await layout('invalid_login_mobile');await shot('08_login_error.png',true);
    await page.getByRole('link',{name:'注册普通账号',exact:true}).click();
    check('registration_form_present',await page.locator('[name=password1]').isVisible());await layout('register_mobile');
    await page.locator('[name=username]').fill('not-valid/name');await page.locator('[name=email]').fill('invalid@example.test');await page.locator('[name=password1]').fill('short');await page.locator('[name=password2]').fill('different');await page.getByRole('button',{name:'注册',exact:true}).click();
    check('registration_server_validation_visible',await page.locator('.errorlist').count()>0);await layout('registration_error_mobile');await shot('09_register_error.png',true);
    await page.getByRole('link',{name:'已有账号，去登录',exact:true}).click();
    await page.getByLabel('账号',{exact:true}).fill(credentials.owner);await page.getByLabel('密码',{exact:true}).fill(credentials.password);await page.getByRole('button',{name:'登录',exact:true}).click();
    check('real_existing_account_login',new URL(page.url()).pathname==='/cinema/personal/');
    await page.getByRole('button',{name:/菜单/}).click();check('mobile_menu_expands',await page.getByRole('button',{name:/菜单/}).getAttribute('aria-expanded')==='true');
    check('mobile_five_clickable_links',await page.getByRole('navigation',{name:'主导航'}).getByRole('link').count()===5);await layout('mobile_menu');await shot('10_mobile_menu.png',true);
    await page.getByRole('navigation').getByRole('link',{name:'首页',exact:true}).click();
    for(const v of widths) {
      await page.setViewportSize(v); await page.goto(base+'/cinema/',{waitUntil:'networkidle'});const l=await layout('home_'+v.width);
      if(v.width>1000)check('desktop_nav_single_row_'+v.width,l.brand.bottom<=l.header.bottom&&l.menu.top<l.brand.bottom&&l.account.top<l.brand.bottom,l);
    }
    await page.setViewportSize(widths[0]);await page.goto(base+'/cinema/',{waitUntil:'networkidle'});await loadedImages('home');await shot('03_home.png');
    await page.getByRole('navigation').getByRole('link',{name:'影片库',exact:true}).click();await loadedImages('catalogue');await layout('catalogue_desktop');
    check('catalogue_cards_visible',await page.locator('article.film-card').count()===12);
    await page.setViewportSize(widths[3]);await layout('catalogue_mobile');await shot('11_catalogue_mobile.png');
    await page.setViewportSize(widths[0]);await page.goto(base+'/cinema/personal/',{waitUntil:'networkidle'});
    const traceLink=page.locator('main a[href^="/cinema/personal/traces/"]').first();check('existing_trace_available',await traceLink.count()===1);await traceLink.click();
    check('top20_twenty_cards',await page.locator('article.film-card').count()===20);check('source_rule_not_ukmap',await page.getByText('普通用户 · 元数据冷启动',{exact:true}).isVisible());
    check('rank_numbers_preserved',(await page.locator('.card-rank').allTextContents()).map(Number).every((n,i)=>n===i+1));await loadedImages('top20');await layout('top20_desktop');await shot('04_recommend_top20.png');
    await page.setViewportSize(widths[3]);await layout('top20_mobile');await shot('12_top20_mobile.png');await page.setViewportSize(widths[0]);
    await page.locator('article.film-card .poster-wrap').first().click();await loadedImages('detail');await layout('detail_desktop');
    check('detail_original_trace_rank_score',await page.locator('#original-rank').innerText()==='1' && (await page.locator('#original-score').innerText()).trim().length>0);
    check('verified_existing_cast_asset',!(await page.locator('#movie-actors').innerText()).includes('未提供')&&!(await page.locator('#movie-directors').innerText()).includes('未提供'));
    await shot('05_movie_detail.png');await page.setViewportSize(widths[3]);await layout('detail_mobile');await shot('13_detail_mobile.png',true);
    await page.setViewportSize(widths[0]);await page.getByRole('navigation').getByRole('link',{name:'研究推荐',exact:true}).click();
    check('ordinary_account_no_research_form',await page.locator('#recommend-form').count()===0);await layout('research_permissions');await shot('14_research_permissions.png');
    const profiles=await page.request.get(base+'/api/research/profiles/');check('ordinary_account_profiles_empty',(await profiles.json()).profiles.length===0);
    const forbidden=await page.request.get(base+'/cinema/personal/traces/'+credentials.other_trace+'/');check('cross_account_trace_denied',forbidden.status()===404);
    await page.getByRole('navigation').getByRole('link',{name:'智能助手',exact:true}).click();check('assistant_actual_boundary_visible',await page.getByText(/未启用Agent/).isVisible());await layout('assistant');
    await page.getByRole('button',{name:'退出',exact:true}).click();await page.goto(base+'/cinema/personal/');check('logout_requires_login',new URL(page.url()).pathname==='/cinema/login/');
    await page.getByLabel('账号',{exact:true}).fill(credentials.owner);await page.getByLabel('密码',{exact:true}).fill(credentials.password);await page.getByRole('button',{name:'登录',exact:true}).click();check('relogin_succeeds',new URL(page.url()).pathname==='/cinema/personal/');
    await page.getByRole('button',{name:'退出',exact:true}).click();
    check('no_frontend_js_exceptions',errors.length===0,errors);check('no_static_404',resources.length>0&&resources.every(r=>r.status<400),resources);
    return {completed:true,checks,layouts,shots,errors,resources,browser:page.context().browser().version(),new_test_access:0};
  } catch(error) {return {completed:false,failure:String(error),checks,layouts,shots,errors,resources};}
}
